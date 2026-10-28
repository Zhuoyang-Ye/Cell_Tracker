"""ROI Alignment & Tracking — Python V2.

Run: python main.py (from this folder).
See README.md for presets, session QC, matching, caching and export conventions.
"""

import sys
from pathlib import Path

# Make sure the package root is importable
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName('ROI Alignment & Tracking')
    app.setStyle('Fusion')

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
