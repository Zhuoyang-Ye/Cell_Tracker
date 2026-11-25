"""Versioned per-pair cache; numeric NPZ files, with no pickle loading."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import sys
import numpy as np

# 3: keys no longer contain the folder path or file times (2026-11-24).
CACHE_VERSION = 3


def default_cache_dir():
    """This computer's per-user cache folder; an app's own folder may be read-only."""
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Caches'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return base / 'Cell_Tracker' / 'alignment_cache'


def fingerprint(session):
    """Identify a session by what registration and IOU read, not where it is stored.

    Leaving out the folder path and file times keeps keys equal for the same data
    on Windows (Z:) and macOS (/Volumes). Changed suite2p files are caught by
    Session.validate, and ops.npy enters only through the image size.
    """
    h = hashlib.sha256()
    h.update(session.name.encode())
    h.update(str(tuple(int(v) for v in session.image_size)).encode())
    for arr in [session.iscell_prob, *session.xpix, *session.ypix]:
        a = np.ascontiguousarray(arr)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def alignment_settings(param):
    defaults = dict(alignment_top_n=200, mask_radius=0, opt_scale=0.5,
                    max_iter=2000, tol_fun=1e-4, tol_x=1e-4,
                    v_lower=[-15, -50, -50, .85, .85, -.15],
                    v_upper=[15, 50, 50, 1.15, 1.15, .15])
    return {k: param.get(k, v) for k, v in defaults.items()}


def cache_key(*items):
    data = json.dumps([CACHE_VERSION, *items], sort_keys=True,
                      default=lambda value: value.tolist())
    return hashlib.sha256(data.encode()).hexdigest()


class PairStore:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else None
        self.hits = 0
        self.warnings = []

    def load(self, key):
        if self.folder is None:
            return None
        path = self.folder / f'{key}.npz'
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                result = {name: z[name] for name in z.files}
            self.hits += 1
            return result
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            self.warnings.append(f'Ignored unreadable cache file: {path.name}')
            return None

    def save(self, key, **values):
        if self.folder is None:
            return
        temp = None
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.folder, suffix='.npz', delete=False) as f:
                temp = Path(f.name)
                np.savez_compressed(f, **values)
            os.replace(temp, self.folder / f'{key}.npz')
        except OSError as exc:
            self.warnings.append(f'Cache could not be saved: {exc}')
        finally:
            if temp is not None and temp.exists():
                try:
                    temp.unlink()
                except OSError as exc:
                    self.warnings.append(f'Cache temporary file could not be removed: {exc}')
