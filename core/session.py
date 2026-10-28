from __future__ import annotations
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def _session_label(folder: Path) -> str:
    """Recognise animal/date/NN and legacy animal-date/NN suite2p layouts."""
    folder = Path(folder)
    if folder.parent.name.lower() == 'suite2p':
        session = folder.parent.parent
        if re.fullmatch(r'A\d+-\d{8}-\d+', session.name):
            return session.name
        if re.fullmatch(r'\d+', session.name):
            num = session.name.zfill(2)
            if re.fullmatch(r'A\d+-\d{8}', session.parent.name):
                return f'{session.parent.name}-{num}'
            if (re.fullmatch(r'\d{8}', session.parent.name)
                    and re.fullmatch(r'A\d+', session.parent.parent.name)):
                return f'{session.parent.parent.name}-{session.parent.name}-{num}'
        return session.name
    return folder.name


def source_stamp(folder):
    """Cheap change detection for already-loaded suite2p files."""
    result = {}
    for name in ('stat.npy', 'iscell.npy', 'ops.npy', 'F.npy', 'Fneu.npy'):
        path = Path(folder) / name
        if path.exists():
            st = path.stat()
            result[name] = (st.st_size, st.st_mtime_ns)
    return result


@dataclass
class Session:
    """One imaging session loaded from a suite2p output folder."""

    name: str
    folder: Path
    xpix: list          # list of np.ndarray int32 — column indices per ROI
    ypix: list          # list of np.ndarray int32 — row indices per ROI
    iscell: np.ndarray  # (n_rois, 2): col0 = is_cell bool, col1 = probability
    mean_img: np.ndarray   # (H, W) float32
    image_size: tuple      # (H, W)
    F: Optional[np.ndarray] = None     # (n_rois, n_frames)
    Fneu: Optional[np.ndarray] = None  # (n_rois, n_frames)
    loaded_stamp: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, folder: str | Path) -> Session:
        """Load from a suite2p plane folder (containing stat.npy, ops.npy, iscell.npy)."""
        folder = Path(folder).resolve()
        stamp = source_stamp(folder)

        ops = np.load(folder / 'ops.npy', allow_pickle=True).item()
        stat = np.load(folder / 'stat.npy', allow_pickle=True)
        iscell = np.load(folder / 'iscell.npy')

        h, w = int(ops['Ly']), int(ops['Lx'])
        mean_img = ops['meanImg'].astype(np.float32)

        xpix = [np.asarray(s['xpix']) for s in stat]
        ypix = [np.asarray(s['ypix']) for s in stat]

        F = np.load(folder / 'F.npy', mmap_mode='r') if (folder / 'F.npy').exists() else None
        Fneu = np.load(folder / 'Fneu.npy', mmap_mode='r') if (folder / 'Fneu.npy').exists() else None

        session = cls(
            name=_session_label(folder),
            folder=folder,
            xpix=xpix,
            ypix=ypix,
            iscell=iscell,
            mean_img=mean_img,
            image_size=(h, w),
            F=F,
            Fneu=Fneu,
            loaded_stamp=stamp,
        )
        session.validate()
        return session

    @property
    def animal_id(self):
        match = re.match(r'^(A\d+)-', self.name)
        return match.group(1) if match else None

    def validate(self):
        """Reject inconsistent ROI identities before registration or trace export."""
        def fail(message):
            raise ValueError(f'{self.name}: {message}')
        if self.loaded_stamp and source_stamp(self.folder) != self.loaded_stamp:
            fail('suite2p files changed; remove and reload this session.')
        n = self.n_rois
        if self.iscell.ndim != 2 or self.iscell.shape[0] != n or self.iscell.shape[1] < 2:
            fail('stat/iscell ROI counts must agree; iscell needs two columns.')
        if not np.all(np.isfinite(self.iscell_prob)) or np.any((self.iscell_prob < 0) | (self.iscell_prob > 1)):
            fail('iscell probabilities must be finite and between 0 and 1.')
        h, w = self.image_size
        if h <= 0 or w <= 0 or self.mean_img.shape != (h, w):
            fail('meanImg dimensions must agree with Ly/Lx.')
        if len(self.ypix) != n:
            fail('xpix/ypix ROI counts differ.')
        for i, (x, y) in enumerate(zip(self.xpix, self.ypix)):
            x, y = np.asarray(x), np.asarray(y)
            if x.ndim != 1 or y.shape != x.shape:
                fail(f'ROI {i}: xpix/ypix must be equal-length vectors.')
            if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                    or np.any(x != np.floor(x)) or np.any(y != np.floor(y))
                    or np.any((x < 0) | (x >= w) | (y < 0) | (y >= h))):
                fail(f'ROI {i}: invalid pixel coordinates.')
        for label, trace in (('F', self.F), ('Fneu', self.Fneu)):
            if trace is not None and (trace.ndim != 2 or trace.shape[0] != n
                                      or trace.dtype.kind not in 'fiu'):
                fail(f'{label} must be a numeric ROI × frame array with {n} rows.')
        if self.F is not None and self.Fneu is not None and self.F.shape != self.Fneu.shape:
            fail('F and Fneu frame counts differ.')

    @property
    def n_rois(self) -> int:
        return len(self.xpix)

    @property
    def iscell_prob(self) -> np.ndarray:
        return self.iscell[:, 1]

    @property
    def iscell_bool(self) -> np.ndarray:
        return self.iscell[:, 0].astype(bool)
