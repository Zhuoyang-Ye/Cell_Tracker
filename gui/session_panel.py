from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QFileDialog, QProgressBar, QCheckBox, QRadioButton, QComboBox, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal


PRESETS = {
    'MATLAB V2 settings': dict(alignment_top_n=200, tracking_prob_threshold=.2,
        iou_threshold=.4, mask_radius=0, opt_scale=.5, max_iter=2000,
        count_threshold=.9, min_cells_cutoff=60, max_missing_sessions=2),
    'Previous Python settings': dict(alignment_top_n=300, tracking_prob_threshold=0.,
        iou_threshold=.3, mask_radius=0, opt_scale=1., max_iter=5000,
        count_threshold=.9, min_cells_cutoff=0, max_missing_sessions=2),
}


class SessionPanel(QWidget):
    session_added = pyqtSignal(str)
    session_removed = pyqtSignal(int)
    session_selected = pyqtSignal(int)
    run_requested = pyqtSignal(dict)
    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(350)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self._groups = []
        sessions = QGroupBox('Sessions — check to include')
        sl = QVBoxLayout(sessions)
        self._list = QListWidget()
        self._list.setMinimumHeight(110)
        self._list.setMaximumHeight(160)
        self._list.currentRowChanged.connect(self.session_selected)
        self._list.itemChanged.connect(lambda _: self.settings_changed.emit())
        sl.addWidget(self._list)
        buttons = QHBoxLayout()
        self._btn_add = QPushButton('Add session…')
        self._btn_rem = QPushButton('Remove')
        self._btn_add.clicked.connect(self._add_session)
        self._btn_rem.clicked.connect(self._remove_session)
        buttons.addWidget(self._btn_add)
        buttons.addWidget(self._btn_rem)
        sl.addLayout(buttons)
        hint = QLabel('Choose …/A811/20260422/01/suite2p/plane0\nOne animal and imaging plane per run.')
        hint.setWordWrap(True)
        sl.addWidget(hint)
        layout.addWidget(sessions)
        self._groups.append(sessions)

        parameters = QGroupBox('Alignment and matching')
        pl = QVBoxLayout(parameters)
        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._preset_combo.setMinimumContentsLength(12)
        self._preset_combo.addItems(list(PRESETS))
        apply = QPushButton('Apply')
        apply.clicked.connect(self._apply_preset)
        preset_row.addWidget(self._preset_combo)
        preset_row.addWidget(apply)
        pl.addLayout(preset_row)
        note = QLabel('Presets fill the controls below; every value can be changed.\n'
                      'Both presets use one-to-one matching and conflict checks.')
        note.setWordWrap(True)
        pl.addWidget(note)
        self._controls = {}
        self._align_top_n = self._number(pl, 'alignment_top_n', 'Align top-N ROIs', 200, 1, 10000,
            'Highest suite2p cell probabilities; these ROIs determine the alignment.')
        self._prob_thresh = self._number(pl, 'tracking_prob_threshold', 'Track probability ≥', .2, 0., 1.,
            'Include ROIs at or above this probability in matching. Zero includes all ROIs.')
        self._iou_thresh = self._number(pl, 'iou_threshold', 'Match IOU ≥', .4, 0., 1.,
            'Required spatial overlap. Larger values require a closer match. Zero overlap never matches.')
        self._mask_radius = self._number(pl, 'mask_radius', 'Mask radius (pixels)', 0, 0, 20,
            'Dilate ROI masks before alignment and overlap calculation.')
        self._opt_scale = self._number(pl, 'opt_scale', 'Search step scale', .5, .01, 5.,
            'Scales the optimizer step sizes, not the image coordinates.')
        self._max_iter = self._number(pl, 'max_iter', 'Max evaluations / start', 2000, 100, 100000,
            'Maximum objective evaluations for each of five optimizer starts.')
        layout.addWidget(parameters)
        self._groups.append(parameters)

        quality = QGroupBox('Session quality and export selection')
        ql = QVBoxLayout(quality)
        self._number(ql, 'count_threshold', 'Quality probability ≥', .9, 0., 1.,
            'Separate from tracking probability: used only to count high-confidence ROIs for session quality.')
        self._number(ql, 'min_cells_cutoff', 'Min high-confidence ROIs', 60, 0, 100000,
            'Exclude sessions below this count before alignment. Set 0 to disable this cutoff. See Session QC.')
        self._number(ql, 'max_missing_sessions', 'Max missing sessions', 2, 0, 10000,
            'For unified export: keep tracks present in at least max(1, included sessions minus this value). '
            'Zero requires presence in all included sessions. Diagnostics still show all unambiguous tracks.')
        label = QLabel('Session QC shows exclusions before running.\nSet minimum count to 0 to retain low-count sessions.')
        label.setWordWrap(True)
        ql.addWidget(label)
        layout.addWidget(quality)
        self._groups.append(quality)

        mode = QGroupBox('Tracking mode')
        ml = QVBoxLayout(mode)
        self._radio_global = QRadioButton('Global — V2, all session pairs')
        self._radio_chain = QRadioButton('Chain — loaded session order')
        self._radio_hub = QRadioButton('Hub — one reference session')
        for radio in (self._radio_global, self._radio_chain, self._radio_hub):
            ml.addWidget(radio)
            radio.toggled.connect(self.settings_changed)
        self._radio_global.setChecked(True)
        self._radio_global.setToolTip('Ambiguous groups containing two ROIs from one session are excluded and reported.')
        self._ref_combo = QComboBox()
        self._ref_combo.addItem('Middle included session (auto)')
        self._ref_combo.setEnabled(False)
        self._ref_combo.currentIndexChanged.connect(self.settings_changed)
        self._radio_hub.toggled.connect(self._ref_combo.setEnabled)
        ml.addWidget(self._ref_combo)
        layout.addWidget(mode)
        self._groups.append(mode)

        self._cache_check = QCheckBox('Reuse disk cache')
        self._cache_check.setChecked(True)
        self._cache_folder = Path(__file__).resolve().parents[1] / '.alignment_cache'
        self._cache_check.setToolTip(f'Cache folder: {self._cache_folder}\nInvalidated when inputs or relevant settings change.')
        outer.addWidget(self._cache_check)
        self._btn_run = QPushButton('Run alignment and tracking')
        self._btn_run.clicked.connect(self._run)
        outer.addWidget(self._btn_run)
        self._progress = QProgressBar()
        self._progress.hide()
        outer.addWidget(self._progress)
        self._status_lbl = QLabel('')
        self._status_lbl.setWordWrap(True)
        outer.addWidget(self._status_lbl)
        layout.addStretch()

    def _number(self, layout, key, label, value, low, high, tip):
        row = QHBoxLayout()
        text = QLabel(label)
        text.setWordWrap(True)
        widget = QDoubleSpinBox() if isinstance(value, float) else QSpinBox()
        widget.setFixedWidth(95)
        if isinstance(value, float):
            widget.setDecimals(3)
            widget.setSingleStep(.05)
        widget.setRange(low, high)
        widget.setValue(value)
        widget.setToolTip(tip)
        text.setToolTip(tip)
        row.addWidget(text)
        row.addWidget(widget)
        layout.addLayout(row)
        self._controls[key] = widget
        widget.valueChanged.connect(self.settings_changed)
        return widget

    def _apply_preset(self):
        for key, value in PRESETS[self._preset_combo.currentText()].items():
            self._controls[key].blockSignals(True)
            self._controls[key].setValue(value)
            self._controls[key].blockSignals(False)
        self.settings_changed.emit()

    def get_param(self):
        result = {key: widget.value() for key, widget in self._controls.items()}
        ref = self._ref_combo.currentIndex()
        result.update(tol_fun=1e-4, tol_x=1e-4,
                      v_lower=[-15, -50, -50, .85, .85, -.15],
                      v_upper=[15, 50, 50, 1.15, 1.15, .15],
                      tracking_mode='hub' if self._radio_hub.isChecked() else
                                    'chain' if self._radio_chain.isChecked() else 'global',
                      hub_ref=None if ref == 0 else ref - 1,
                      use_disk_cache=self._cache_check.isChecked(), cache_dir=str(self._cache_folder),
                      exclude_paths=[self._list.item(i).data(Qt.UserRole) for i in range(self._list.count())
                                     if self._list.item(i).checkState() != Qt.Checked])
        return result

    def update_ref_combo(self, names):
        self._ref_combo.blockSignals(True)
        self._ref_combo.clear()
        self._ref_combo.addItem('Middle included session (auto)')
        self._ref_combo.addItems(names)
        self._ref_combo.blockSignals(False)

    def add_session_item(self, name, n_rois, folder):
        self._list.blockSignals(True)
        item = QListWidgetItem(f'{name} ({n_rois} ROIs)')
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setData(Qt.UserRole, str(folder))
        item.setCheckState(Qt.Checked)
        item.setToolTip(str(folder))
        self._list.addItem(item)
        self._list.blockSignals(False)

    def remove_session_item(self, row):
        self._list.takeItem(row)

    def _add_session(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select suite2p plane folder (stat.npy, ops.npy, iscell.npy)')
        if folder:
            self.session_added.emit(folder)

    def _remove_session(self):
        if self._list.currentRow() >= 0:
            self.session_removed.emit(self._list.currentRow())

    def _run(self):
        if self._btn_run.isEnabled():
            self.run_requested.emit(self.get_param())

    def set_running(self, running):
        for group in self._groups:
            group.setEnabled(not running)
        self._btn_run.setEnabled(not running)
        self._cache_check.setEnabled(not running)
        self._progress.setVisible(running)

    def set_progress(self, value, maximum, text=''):
        self._progress.setMaximum(maximum)
        self._progress.setValue(value)
        if text:
            self.set_status(text)

    def set_status(self, text):
        self._status_lbl.setText(text)
