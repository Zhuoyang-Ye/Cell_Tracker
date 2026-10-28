"""Offscreen GUI integration checks; no desktop interaction or real dataset writes."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'align_roi_test_mpl'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from gui.main_window import MainWindow
from test_v2 import session, make_cache
from core.tracking import apply_tracking


class GuiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.window = MainWindow()
        self.window._sess_panel._cache_folder = Path(self.temp.name)/'cache'
        self.window._sess_panel._controls['min_cells_cutoff'].setValue(0)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.temp.cleanup()

    def wait_for_work(self):
        end = time.monotonic()+20
        while self.window._is_running() and time.monotonic() < end:
            QTest.qWait(10)
        QTest.qWait(20)
        self.assertFalse(self.window._is_running(), 'Background worker did not finish')

    def load_sessions(self):
        for i in range(3):
            s = session(f'A811-2026042{i+1}-01', folder=Path(self.temp.name)/f'session{i}')
            with patch('gui.main_window.Session.load', return_value=s):
                self.window._on_session_added(str(s.folder))
        self.app.processEvents()

    def test_worker_qc_exclusion_cache_and_export(self):
        self.load_sessions()
        panel = self.window._sess_panel
        self.assertEqual(self.window._results._qc_table.rowCount(), 3)
        panel._list.item(0).setCheckState(Qt.Unchecked)
        self.assertIn('Excluded', self.window._results._qc_table.item(0,6).text())
        panel._radio_hub.setChecked(True)
        panel._ref_combo.setCurrentIndex(3)  # original session 2 -> kept index 1
        with patch('gui.main_window.QMessageBox.critical') as error:
            self.window._run_batch(panel.get_param())
            self.wait_for_work()
            self.assertFalse(error.called, str(error.call_args))
        self.assertEqual(self.window._result.param['hub_ref'], 1)
        self.assertEqual(self.window._result.tracking_matrix.shape, (2,2))
        cache = self.window._pairwise_cache
        panel._iou_thresh.setValue(.5)
        self.assertIsNone(self.window._result)
        self.assertFalse(self.window._results._btn_export.isEnabled())
        self.window._run_batch(panel.get_param())
        self.wait_for_work()
        self.assertIs(cache, self.window._pairwise_cache)
        self.window._results._mat_check.setChecked(True)
        with patch('gui.main_window.QFileDialog.getExistingDirectory', return_value=self.temp.name), \
             patch('gui.main_window.QMessageBox.information'), \
             patch('gui.main_window.QMessageBox.critical') as error:
            self.window._export_results()
            self.wait_for_work()
            self.assertFalse(error.called, str(error.call_args))
        outputs = list(Path(self.temp.name).glob('roi_tracking_*'))
        self.assertEqual(len(outputs), 1)
        self.assertTrue((outputs[0]/'manifest.json').exists())
        self.assertEqual(len(list(outputs[0].glob('*_Fu.mat'))), 2)

    def test_presets_and_reference_guard(self):
        self.load_sessions()
        panel = self.window._sess_panel
        panel._preset_combo.setCurrentText('Previous Python settings')
        panel._apply_preset()
        self.assertEqual(panel.get_param()['tracking_prob_threshold'], 0)
        panel._preset_combo.setCurrentText('MATLAB V2 settings')
        panel._apply_preset()
        self.assertEqual(panel.get_param()['min_cells_cutoff'], 60)
        self.assertEqual(panel.get_param()['tracking_prob_threshold'], .2)
        self.assertTrue(all('Excluded' in self.window._results._qc_table.item(i,6).text() for i in range(3)))
        panel._controls['min_cells_cutoff'].setValue(0)
        panel._radio_hub.setChecked(True)
        panel._ref_combo.setCurrentIndex(1)
        panel._list.item(0).setCheckState(Qt.Unchecked)
        with patch('gui.main_window.QMessageBox.information') as notice:
            self.window._run_batch(panel.get_param())
            self.assertIn('reference is excluded', str(notice.call_args))
        self.assertIsNone(self.window._worker)

    def test_conflict_only_results_render_and_clear(self):
        cache = make_cache([2,1,1], {(0,1):[[.8],[.1]], (1,2):[[.8]], (0,2):[[.1],[.8]]})
        r = apply_tracking(cache, {})
        panel = self.window._results
        panel.set_result(r, [s.name for s in r.sessions])
        self.app.processEvents()
        self.assertIn('1 ambiguous', panel._export_label.text())
        self.assertIn('ROI 1', panel._stats_edit.toPlainText())
        panel.clear()
        panel.set_result(r, [s.name for s in r.sessions])
        self.app.processEvents()
        self.assertEqual(len(panel._shared_canvas.fig.axes), 2)  # no accumulating colorbars


if __name__ == '__main__':
    unittest.main()
