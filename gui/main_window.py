from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QMessageBox, QFileDialog, QProgressBar,
    QStatusBar, QAction, QToolBar, QRadioButton, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .image_canvas import ImageCanvas
from .session_panel import SessionPanel
from .results_panel import ResultsPanel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.session import Session
from core.quality import prepare_sessions, session_report
from core.export import export_results
from core.tracking import (BatchResult, PairwiseCache,
                           compute_pairwise, apply_tracking, cache_is_valid)


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _BatchWorker(QThread):
    pair_progress  = pyqtSignal(int, int, str)
    step_progress  = pyqtSignal(str, int, int)
    cache_ready    = pyqtSignal(object)   # PairwiseCache — save in main window
    finished       = pyqtSignal(object)   # BatchResult
    error          = pyqtSignal(str)

    def __init__(self, sessions: list, param: dict,
                 existing_cache: PairwiseCache | None = None):
        super().__init__()
        self.sessions       = sessions
        self.param          = param
        self.existing_cache = existing_cache

    def run(self):
        try:
            kept, report = prepare_sessions(self.sessions, self.param)
            param = dict(self.param)
            if param.get('tracking_mode') == 'hub' and param.get('hub_ref') is not None:
                reference = self.sessions[int(param['hub_ref'])]
                param['hub_ref'] = next(i for i, session in enumerate(kept) if session is reference)
            if cache_is_valid(self.existing_cache, kept, param):
                cache = self.existing_cache
            else:
                def pair_progress(current, total, label):
                    if self.isInterruptionRequested():
                        raise InterruptedError('Run cancelled.')
                    self.pair_progress.emit(current, total, label)
                def step_progress(step, current, total):
                    if self.isInterruptionRequested():
                        raise InterruptedError('Run cancelled.')
                    self.step_progress.emit(step, current, total)
                cache = compute_pairwise(kept, param, pair_progress, step_progress)
                self.cache_ready.emit(cache)
            if self.isInterruptionRequested():
                raise InterruptedError('Run cancelled.')
            result = apply_tracking(cache, param)
            result.session_qc = report
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _ExportWorker(QThread):
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, result, folder, options):
        super().__init__()
        self.result, self.folder, self.options = result, folder, options

    def run(self):
        try:
            out = export_results(self.result, self.folder, **self.options)
            self.completed.emit(str(out))
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ROI Alignment & Tracking — V2')
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1400, screen.width() - 40), min(800, screen.height() - 60))

        self._sessions: list[Session] = []
        self._result: BatchResult | None = None
        self._worker: _BatchWorker | None = None
        self._pairwise_cache: PairwiseCache | None = None

        self._export_worker = None
        self._closing = False

        self._build_ui()
        self._connect_signals()
        self.statusBar().showMessage('Ready — add sessions with the left panel.')

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left: session panel
        self._sess_panel = SessionPanel()
        splitter.addWidget(self._sess_panel)

        # Centre: session selector + FOV controls + image canvas(es)
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(4, 4, 4, 4)
        cv.setSpacing(4)

        # Row 1: session selector + overlay combo
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel('View session:'))
        self._sess_combo = QComboBox()
        self._sess_combo.setMinimumWidth(200)
        self._sess_combo.setToolTip('Select which loaded session to display in the canvas.')
        top_row.addWidget(self._sess_combo)
        top_row.addStretch()
        top_row.addWidget(QLabel('Overlay:'))
        self._overlay_combo = QComboBox()
        self._overlay_combo.addItems(['ROI overlay', 'No overlay (FOV only)'])
        self._overlay_combo.setToolTip(
            'ROI overlay — show coloured ROI masks on the mean image.\n'
            'No overlay (FOV only) — hide all ROI masks; display the raw mean image.'
        )
        top_row.addWidget(self._overlay_combo)
        cv.addLayout(top_row)

        # Row 2: FOV display mode (single vs split)
        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel('FOV display:'))
        self._radio_single = QRadioButton('Single view  (all ROIs)')
        self._radio_dual   = QRadioButton('Split view  (Cells | Non-cells)')
        self._radio_single.setChecked(True)
        self._radio_single.setToolTip(
            'Show all ROIs in one canvas.\n'
            'Green = cells (iscell = True)   Red = non-cells (iscell = False)')
        self._radio_dual.setToolTip(
            'Show two canvases side by side:\n'
            '  Left  — cells only  (iscell = True,  green)\n'
            '  Right — non-cells only  (iscell = False, red)\n\n'
            'Useful for quickly reviewing suite2p cell classification.')
        fov_row.addWidget(self._radio_single)
        fov_row.addWidget(self._radio_dual)
        fov_row.addStretch()
        cv.addLayout(fov_row)

        # Row 3: colour legend — always visible
        legend = QLabel(
            'Each ROI has a unique random colour&nbsp;&nbsp;&nbsp;'
            '<span style="color:#777;">(non-cells shown at reduced opacity in single view)</span>'
            '&nbsp;&nbsp;&nbsp;'
            '<span style="color:#ffff00; font-size:15px;">■</span>'
            ' Yellow = selected ROI&nbsp;&nbsp;&nbsp;'
            '<span style="color:#777;">Scroll = zoom&nbsp;·&nbsp;'
            'Click = select&nbsp;·&nbsp;Double-click = preview classification&nbsp;·&nbsp;'
            'Space = toggle ROI overlay&nbsp;·&nbsp;'
            '1 = single view&nbsp;·&nbsp;2 = split view</span>'
        )
        legend.setStyleSheet('font-size: 13px; padding: 1px 0;')
        legend.setWordWrap(True)
        cv.addWidget(legend)

        # Canvas area: left pane (always shown) + right pane (split mode only)
        canvas_area = QWidget()
        canvas_hl = QHBoxLayout(canvas_area)
        canvas_hl.setContentsMargins(0, 0, 0, 0)
        canvas_hl.setSpacing(4)

        left_pane = QWidget()
        left_vl = QVBoxLayout(left_pane)
        left_vl.setContentsMargins(0, 0, 0, 0)
        left_vl.setSpacing(2)
        self._left_title = QLabel('Cells  (iscell = True)')
        self._left_title.setAlignment(Qt.AlignCenter)
        self._left_title.setStyleSheet('font-weight: bold; color: #50c832;')
        self._left_title.setVisible(False)
        self._canvas = ImageCanvas()
        left_vl.addWidget(self._left_title)
        left_vl.addWidget(self._canvas)
        canvas_hl.addWidget(left_pane)

        self._right_pane = QWidget()
        right_vl = QVBoxLayout(self._right_pane)
        right_vl.setContentsMargins(0, 0, 0, 0)
        right_vl.setSpacing(2)
        right_title = QLabel('Non-cells  (iscell = False)')
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setStyleSheet('font-weight: bold; color: #c83250;')
        self._canvas_right = ImageCanvas()
        right_vl.addWidget(right_title)
        right_vl.addWidget(self._canvas_right)
        self._right_pane.setVisible(False)
        canvas_hl.addWidget(self._right_pane)

        cv.addWidget(canvas_area)

        # ROI info label below canvas
        self._roi_lbl = QLabel(
            'Click an ROI for info. Double-click previews classification; tracking uses probability.')
        self._roi_lbl.setStyleSheet('color: #888; font-size: 13px;')
        self._roi_lbl.setWordWrap(True)
        cv.addWidget(self._roi_lbl)

        splitter.addWidget(centre)

        # Right: results panel
        self._results = ResultsPanel()
        splitter.addWidget(self._results)

        splitter.setSizes([350, 550, 500])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        self.setCentralWidget(splitter)

        # Menu bar
        menu = self.menuBar()
        file_menu = menu.addMenu('File')
        act_add = file_menu.addAction('Add session…')
        act_add.triggered.connect(self._sess_panel._add_session)
        file_menu.addSeparator()
        act_export = file_menu.addAction('Export results…')
        act_export.triggered.connect(self._export_results)
        file_menu.addSeparator()
        act_quit = file_menu.addAction('Quit')
        act_quit.triggered.connect(self.close)

        run_menu = menu.addMenu('Run')
        act_run = run_menu.addAction('Run batch alignment')
        act_run.triggered.connect(lambda: self._sess_panel._run())

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        sp = self._sess_panel
        sp.session_added.connect(self._on_session_added)
        sp.session_removed.connect(self._on_session_removed)
        sp.session_selected.connect(self._on_session_selected)
        sp.run_requested.connect(self._run_batch)
        sp.settings_changed.connect(self._on_settings_changed)

        self._sess_combo.currentIndexChanged.connect(self._on_session_combo)
        self._overlay_combo.currentIndexChanged.connect(self._on_overlay_changed)
        self._canvas.roi_clicked.connect(self._on_roi_clicked)
        self._canvas_right.roi_clicked.connect(self._on_roi_clicked)
        self._canvas.roi_toggled.connect(
            lambda i: self._on_roi_toggled(i, self._canvas_right))
        self._canvas_right.roi_toggled.connect(
            lambda i: self._on_roi_toggled(i, self._canvas))
        self._radio_single.toggled.connect(self._on_fov_mode_changed)
        self._results.export_requested.connect(self._export_results)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _on_session_added(self, folder: str):
        if self._is_running():
            return
        if any(s.folder == Path(folder).resolve() for s in self._sessions):
            self.statusBar().showMessage('This session is already loaded.')
            return
        try:
            sess = Session.load(folder)
        except Exception as exc:
            QMessageBox.critical(self, 'Load error', str(exc))
            return

        self._sessions.append(sess)
        self._sess_panel.add_session_item(sess.name, sess.n_rois, sess.folder)
        self._sess_combo.addItem(sess.name)
        self._sess_panel.update_ref_combo([s.name for s in self._sessions])

        # Auto-select new session
        idx = len(self._sessions) - 1
        self._sess_panel._list.setCurrentRow(idx)
        self._sess_combo.setCurrentIndex(idx)
        self._update_canvas_display(sess)
        self._on_settings_changed()
        self.statusBar().showMessage(
            f'Loaded {sess.name}: {sess.n_rois} ROIs  |  image {sess.image_size}')

    def _on_session_removed(self, row: int):
        if self._is_running():
            return
        if 0 <= row < len(self._sessions):
            self._sessions.pop(row)
            self._sess_panel.remove_session_item(row)
            self._sess_combo.removeItem(row)
            self._sess_panel.update_ref_combo([s.name for s in self._sessions])
            self._on_settings_changed()

    def _on_session_selected(self, row: int):
        if 0 <= row < len(self._sessions):
            self._sess_combo.setCurrentIndex(row)

    def _on_session_combo(self, idx: int):
        if 0 <= idx < len(self._sessions):
            self._update_canvas_display(self._sessions[idx])
            self._sess_panel._list.setCurrentRow(idx)

    # ------------------------------------------------------------------
    # FOV display mode
    # ------------------------------------------------------------------

    def _on_overlay_changed(self) -> None:
        visible = self._overlay_combo.currentIndex() == 0  # 0 = 'ROI overlay'
        self._canvas.set_overlay(visible)
        self._canvas_right.set_overlay(visible)

    def _on_fov_mode_changed(self) -> None:
        dual = self._radio_dual.isChecked()
        self._right_pane.setVisible(dual)
        self._left_title.setVisible(dual)
        self._update_canvas_display()

    def _update_canvas_display(self, sess=None) -> None:
        """Render sess (or the currently selected session) on the canvas(es)."""
        if sess is None:
            idx = self._sess_combo.currentIndex()
            if idx < 0 or idx >= len(self._sessions):
                return
            sess = self._sessions[idx]
        dual = self._radio_dual.isChecked()
        self._canvas.set_filter('cells' if dual else 'all')
        self._canvas.set_session(sess)
        if dual:
            self._canvas_right.set_filter('noncells')
            self._canvas_right.set_session(sess)
            # Share the accepted array so double-click toggles stay in sync
            self._canvas_right._accepted = self._canvas._accepted

    def _on_roi_toggled(self, roi_idx: int, other: 'ImageCanvas') -> None:
        """Refresh the peer canvas after a toggle so both stay in sync."""
        if other._accepted is not None and 0 <= roi_idx < len(other._accepted):
            other._refresh()

    # ------------------------------------------------------------------
    # ROI interaction
    # ------------------------------------------------------------------

    def _on_roi_clicked(self, roi_idx: int):
        view_idx = self._sess_combo.currentIndex()
        if view_idx < 0 or view_idx >= len(self._sessions):
            return
        sess = self._sessions[view_idx]
        prob = float(sess.iscell_prob[roi_idx])
        n_px = len(sess.xpix[roi_idx])
        accepted = bool(self._canvas._accepted[roi_idx]) if self._canvas._accepted is not None else '?'
        self._roi_lbl.setText(
            f'ROI {roi_idx}  |  iscell prob: {prob:.3f}  |  pixels: {n_px}'
            f'  |  accepted: {accepted}'
            f'  |  classification preview only'
        )

    # ------------------------------------------------------------------
    # Batch alignment
    # ------------------------------------------------------------------

    def _is_running(self):
        return ((self._worker is not None and self._worker.isRunning()) or
                (self._export_worker is not None and self._export_worker.isRunning()))

    def _on_settings_changed(self):
        self._result = None
        self._results.clear()
        param = self._sess_panel.get_param()
        report = session_report(self._sessions, param)
        self._results.set_session_report(report, param)
        count = sum(row['included'] for row in report)
        self._sess_panel.set_status(f'{count}/{len(report)} sessions included. See Session QC.')

    def _run_batch(self, param: dict):
        if self._is_running():
            return
        try:
            prepare_sessions(self._sessions, param)
        except ValueError as exc:
            QMessageBox.information(self, 'Session selection', str(exc))
            return
        self._result = None
        self._results.clear()
        self._sess_panel.set_running(True)
        self._sess_panel.set_status('Loading cached pairs or computing alignment…')
        self._worker = _BatchWorker(list(self._sessions), dict(param), self._pairwise_cache)
        self._worker.pair_progress.connect(self._on_pair_progress)
        self._worker.step_progress.connect(self._on_step_progress)
        self._worker.cache_ready.connect(self._on_cache_ready)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.error.connect(self._on_batch_error)
        self._worker.start()

    def _on_pair_progress(self, current: int, total: int, label: str):
        self._sess_panel.set_progress(current, total, f'Pair {current}/{total}: {label}')
        self.statusBar().showMessage(f'Pair {current}/{total}: {label}')

    def _on_step_progress(self, step: str, current: int, total: int):
        step_names = {'alignment': 'Finding alignment', 'iou': 'Computing IOU',
                      'applying': 'Applying transform'}
        msg = step_names.get(step, step)
        if total > 0:
            msg += f' ({current}/{total})'
        self._sess_panel.set_status(msg)

    def _on_cache_ready(self, cache: PairwiseCache):
        self._pairwise_cache = cache

    def _on_batch_finished(self, result: BatchResult):
        self._result = result
        n_sessions = len(result.sessions)
        n_unique   = result.tracking_matrix.shape[0]
        n_all      = int((result.presence_count == n_sessions).sum())
        self._sess_panel.set_running(False)
        self._sess_panel.set_status(
            f'Done.  {n_unique} unique ROIs  |  {n_all} in all {n_sessions} sessions.')
        self.statusBar().showMessage(
            f'Batch complete — {n_unique} unique ROIs tracked, '
            f'{n_all} present in all {n_sessions} sessions.')

        names = [s.name for s in result.sessions]
        self._results.set_result(result, names)
        if self._closing:
            self._worker.wait()
            self.close()

    def _on_batch_error(self, msg: str):
        self._sess_panel.set_running(False)
        self._sess_panel.set_status('Error: ' + msg)
        if self._closing:
            self._worker.wait()
            self.close()
        else:
            QMessageBox.critical(self, 'Batch error', msg)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self._overlay_combo.setCurrentIndex(
                1 - self._overlay_combo.currentIndex()
            )
        elif key == Qt.Key_1:
            self._radio_single.setChecked(True)
        elif key == Qt.Key_2:
            self._radio_dual.setChecked(True)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_results(self):
        if self._is_running():
            return
        if self._result is None:
            QMessageBox.information(self, 'No results', 'Run alignment and tracking first.')
            return
        folder = QFileDialog.getExistingDirectory(self, 'Choose output parent folder')
        if not folder:
            return
        self._sess_panel.set_running(True)
        self._results._btn_export.setEnabled(False)
        self.statusBar().showMessage('Exporting into a new results folder…')
        self._export_worker = _ExportWorker(self._result, folder, self._results.export_options())
        self._export_worker.completed.connect(self._on_export_complete)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_complete(self, folder):
        self._sess_panel.set_running(False)
        self._results._btn_export.setEnabled(True)
        self.statusBar().showMessage(f'Results exported to {folder}')
        if self._closing:
            self._export_worker.wait()
            self.close()
        else:
            QMessageBox.information(self, 'Export complete', f'Results saved to:\n{folder}')

    def _on_export_error(self, message):
        self._sess_panel.set_running(False)
        self._results._btn_export.setEnabled(True)
        if self._closing:
            self._export_worker.wait()
            self.close()
        else:
            QMessageBox.critical(self, 'Export error', message)

    def closeEvent(self, event):
        if self._is_running():
            self._closing = True
            if self._worker and self._worker.isRunning():
                self._worker.requestInterruption()
            self.statusBar().showMessage('Finishing the current operation before closing…')
            event.ignore()
        else:
            event.accept()
