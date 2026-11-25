"""Figure style for the embedded result plots.

Inside the 2P imaging repo (this folder at 2_Img_Python/Cell_Tracker/python), the
lab's shared manuscript style, Utilities/paper_style.py, is used. In a standalone
clone or a packaged app that module is absent, so plain matplotlib is used with
the same font size and plot box. The plots only appear in the GUI, so the
difference is cosmetic.
"""
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt

_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
if (_PIPELINE_ROOT / 'Utilities' / 'paper_style.py').exists():
    sys.path.append(str(_PIPELINE_ROOT))
try:
    from Utilities import paper_style as _ps
except ImportError:
    _ps = None


def use_style(font_size):
    if _ps is not None:
        _ps.use_paper_style(font_size=font_size)
        return
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def subplots(axes_in):
    """One plot box of about ``axes_in`` inches; the canvas calls tight_layout."""
    if _ps is not None:
        return _ps.subplots(axes_in=axes_in)
    width, height = axes_in
    return plt.subplots(figsize=(width + 1.0, height + 0.8))
