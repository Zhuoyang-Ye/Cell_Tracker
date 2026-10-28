from pathlib import Path
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton, QTextEdit,
    QLabel, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
)
from PyQt5.QtCore import pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# The shared manuscript style lives in the parent imaging pipeline.
sys.path.append(str(Path(__file__).resolve().parents[3]))
from Utilities import paper_style as ps
import matplotlib.pyplot as plt


class _MplCanvas(FigureCanvasQTAgg):
    def __init__(self, axes_in=(3.2, 2.6)):
        ps.use_paper_style(font_size=9)
        self.fig, self.ax = ps.subplots(axes_in=axes_in)
        plt.close(self.fig)  # embedded canvas owns this figure, not a separate window
        super().__init__(self.fig)

    def refresh(self):
        self.fig.tight_layout()
        self.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


class ResultsPanel(QWidget):
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(370)
        self._result = None
        self._session_names = []
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.setUsesScrollButtons(True)
        self._qc_table = QTableWidget(0, 7)
        self._qc_table.setHorizontalHeaderLabels(['Session', 'Total', 'Quality', 'Tracking', 'F frames', 'Fneu frames', 'Status'])
        self._qc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tabs.addTab(self._qc_table, 'Session QC')
        self._heatmap_canvas = _MplCanvas()
        self._tabs.addTab(self._heatmap_canvas, 'Tracking')
        self._shared_canvas = _MplCanvas()
        self._tabs.addTab(self._shared_canvas, 'Shared cells')
        self._presence_canvas = _MplCanvas()
        self._tabs.addTab(self._presence_canvas, 'Presence')
        self._shared_colorbar = None

        pair_tab = QWidget()
        pl = QVBoxLayout(pair_tab)
        row = QHBoxLayout()
        self._pair_a, self._pair_b = QComboBox(), QComboBox()
        row.addWidget(QLabel('Sessions:'))
        row.addWidget(self._pair_a)
        row.addWidget(QLabel('↔'))
        row.addWidget(self._pair_b)
        pl.addLayout(row)
        self._iou_canvas = _MplCanvas()
        pl.addWidget(self._iou_canvas)
        for cb in (self._pair_a, self._pair_b):
            cb.currentIndexChanged.connect(self._update_iou_plot)
        self._tabs.addTab(pair_tab, 'IOU')
        self._stats_edit = QTextEdit()
        self._stats_edit.setReadOnly(True)
        self._tabs.addTab(self._stats_edit, 'Stats / conflicts')
        layout.addWidget(self._tabs)
        self._export_label = QLabel('Run tracking to preview the unified export.')
        self._export_label.setWordWrap(True)
        layout.addWidget(self._export_label)
        self._trace_check = QCheckBox('Include unified raw F / Fneu')
        self._trace_check.setChecked(True)
        layout.addWidget(self._trace_check)
        self._mat_check = QCheckBox('Also save MATLAB .mat files')
        self._mat_check.setToolTip('MATLAB cellID uses 1-based original ROI indices and NaN for absent cells.')
        layout.addWidget(self._mat_check)
        self._btn_export = QPushButton('Export results…')
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self.export_requested)
        layout.addWidget(self._btn_export)

    def set_session_report(self, rows, param):
        self._qc_table.setRowCount(len(rows))
        self._qc_table.setHorizontalHeaderLabels(['Session', 'Total',
            f'Prob ≥ {param.get("count_threshold", .9):g}',
            f'Track ≥ {param.get("tracking_prob_threshold", .2):g}', 'F frames', 'Fneu frames', 'Status'])
        for i, row in enumerate(rows):
            status = 'Included' if row['included'] else 'Excluded: ' + row['reason']
            if row['F_frames'] is None:
                status += '; F missing (mapping only)'
            values = [row['session'], row['n_total'], row['n_count'], row['n_track'],
                      row['F_frames'], row['Fneu_frames'], status]
            for j, value in enumerate(values):
                item = QTableWidgetItem('Missing' if value is None else str(value))
                item.setToolTip(row['folder'] if j == 0 else str(value))
                self._qc_table.setItem(i, j, item)
        self._qc_table.resizeColumnsToContents()

    def clear(self):
        self._result = None
        if self._shared_colorbar is not None:
            self._shared_colorbar.remove()
            self._shared_colorbar = None
        for canvas in (self._heatmap_canvas, self._shared_canvas, self._presence_canvas, self._iou_canvas):
            canvas.ax.clear()
            canvas.draw_idle()
        self._stats_edit.clear()
        self._btn_export.setEnabled(False)
        self._export_label.setText('Settings or sessions changed. Run tracking to refresh results.')

    def set_result(self, result, session_names):
        self._result = result
        self._session_names = session_names
        self.set_session_report(result.session_qc, result.param)
        self._populate_tracking_heatmap()
        self._populate_diagnostics()
        self._populate_pair_combos()
        self._populate_stats()
        n = len(result.sessions)
        minimum = max(1, n-result.param.get('max_missing_sessions', 2))
        self._export_label.setText(f'Unified export: {len(result.export_rows)} tracks present in at least '
                                   f'{minimum}/{n} included sessions.\n'
                                   f'{len(result.conflicts)} ambiguous groups excluded.')
        available = all(s.F is not None for s in result.sessions)
        self._trace_check.setEnabled(available)
        self._trace_check.setChecked(available)
        self._trace_check.setToolTip('All included sessions need F.npy for trace export.' if not available else
                                     'Raw fluorescence, with identical cell row order and NaNs for absent cells.')
        self._btn_export.setEnabled(True)
        self._tabs.setCurrentIndex(1)

    def export_options(self):
        return dict(include_traces=self._trace_check.isChecked(), matlab=self._mat_check.isChecked())

    def _populate_tracking_heatmap(self):
        r = self._result
        ax = self._heatmap_canvas.ax
        ax.clear()
        if len(r.tracking_matrix):
            ax.imshow(r.tracking_matrix < 0, cmap='gray', vmin=0, vmax=1,
                      aspect='auto', interpolation='nearest')
        else:
            ax.text(.5, .5, 'No unambiguous tracks', ha='center', transform=ax.transAxes)
        ax.set_xticks(range(len(r.sessions)))
        ax.set_xticklabels(self._session_names, rotation=45, ha='right')
        ax.set_ylabel('Tracked ROI (most present first)')
        ax.set_title('Black = present; white = absent')
        self._heatmap_canvas.refresh()

    def _populate_diagnostics(self):
        r = self._result
        n = len(r.sessions)
        if self._shared_colorbar is not None:
            self._shared_colorbar.remove()
        ax = self._shared_canvas.ax
        ax.clear()
        im = ax.imshow(r.shared_cells, cmap='Blues', vmin=0)
        self._shared_colorbar = self._shared_canvas.fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self._session_names, rotation=45, ha='right')
        ax.set_yticklabels(self._session_names)
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(r.shared_cells[i, j]), ha='center', va='center',
                            color='white' if r.shared_cells[i, j] > r.shared_cells.max()/2 else 'black')
        ax.set_title('One-to-one shared cells\nBefore global conflict rejection')
        self._shared_canvas.refresh()
        ax = self._presence_canvas.ax
        ax.clear()
        counts = np.bincount(r.presence_count, minlength=n+1)[1:n+1]
        ax.bar(np.arange(1, n+1), counts, color='steelblue')
        minimum = max(1, n-r.param.get('max_missing_sessions', 2))
        ax.axvline(minimum-.5, color='darkred', linestyle='--', label=f'Export: ≥ {minimum} sessions')
        ax.set_xticks(range(1, n+1))
        ax.set_xlabel('Number of included sessions present')
        ax.set_ylabel('Unambiguous tracks')
        ax.set_title('Presence distribution')
        ax.legend()
        self._presence_canvas.refresh()

    def _populate_pair_combos(self):
        for cb in (self._pair_a, self._pair_b):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(self._session_names)
            cb.blockSignals(False)
        self._pair_b.setCurrentIndex(min(1, len(self._session_names)-1))
        self._update_iou_plot()

    def _update_iou_plot(self):
        if self._result is None:
            return
        i, j = sorted((self._pair_a.currentIndex(), self._pair_b.currentIndex()))
        pr = self._result.pairs.get((i, j))
        if pr is None:
            return
        ax = self._iou_canvas.ax
        ax.clear()
        values = pr.iou_matrix[pr.iou_matrix > 0]
        if len(values):
            ax.hist(values, bins=40, color='steelblue')
            ax.set_yscale('log')
        else:
            ax.text(.5, .5, 'No overlapping ROIs', ha='center', transform=ax.transAxes)
        ax.axvline(self._result.param.get('iou_threshold', .4), color='red', linestyle='--')
        ax.set_xlabel('IOU (all positive candidate overlaps)')
        ax.set_ylabel('Count')
        ax.set_title(f'Registered {self._session_names[i]} → {self._session_names[j]}\n'
                     f'{len(pr.matched_a)} one-to-one matches; {pr.collisions} best-match collisions')
        self._iou_canvas.refresh()

    def _populate_stats(self):
        r = self._result
        n = len(r.sessions)
        total_collisions = sum(pr.collisions for (i, j), pr in r.pairs.items() if i < j)
        lines = [f'Included sessions: {n}', f'Unambiguous tracks: {len(r.tracking_matrix)}',
                 f'Pairwise best-match collisions: {total_collisions}',
                 f'Global ambiguous groups excluded: {len(r.conflicts)}',
                 f'Tracks selected for unified export: {len(r.export_rows)}', '',
                 'Presence (ambiguous groups excluded):']
        for k in range(n, 0, -1):
            lines.append(f'  ≥ {k}/{n}: {int((r.presence_count >= k).sum())}; '
                         f'exactly {k}/{n}: {int((r.presence_count == k).sum())}')
        lines += ['', 'Pair matches / alignment overlap:']
        for (i, j), pr in r.pairs.items():
            if i < j:
                lines.append(f'  {r.sessions[i].name} → {r.sessions[j].name}: '
                             f'{len(pr.matched_a)} matches, {pr.overlap_px} pixels, '
                             f'{pr.collisions} best-match collisions')
        lines += ['', 'Excluded ambiguous groups (original ROI IDs, 0-based):']
        for conflict in r.conflicts:
            lines.append('  ' + '; '.join(f'{r.sessions[m["session"]].name}: ROI {m["original_roi"]}'
                                         for m in conflict['members']))
        if not r.conflicts:
            lines.append('  None')
        lines += ['', 'Cache: ' + str(r.cache_info)]
        self._stats_edit.setPlainText('\n'.join(lines))

