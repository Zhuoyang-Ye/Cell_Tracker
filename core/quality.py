"""Session selection, shared by the GUI and the batch API."""
from pathlib import Path
import numpy as np


def session_report(sessions, param):
    excluded = {str(Path(p).resolve()) for p in param.get('exclude_paths', [])}
    rows = []
    for s in sessions:
        reasons = []
        try:
            s.validate()
        except ValueError as exc:
            reasons.append(str(exc))
        prob = s.iscell[:, 1] if s.iscell.ndim == 2 and s.iscell.shape[1] >= 2 else np.array([])
        count = int(np.sum(prob >= param.get('count_threshold', 0.9)))
        track = int(np.sum(prob >= param.get('tracking_prob_threshold', 0.2)))
        if str(s.folder.resolve()) in excluded:
            reasons.append('manually excluded')
        if count < param.get('min_cells_cutoff', 0):
            reasons.append('below minimum high-confidence count')
        if track == 0:
            reasons.append('no tracking ROIs')
        rows.append(dict(session=s.name, folder=str(s.folder), n_total=s.n_rois,
                         n_count=count, n_track=track,
                         F_frames=None if s.F is None else int(s.F.shape[-1]),
                         Fneu_frames=None if s.Fneu is None else int(s.Fneu.shape[-1]),
                         included=not reasons, reason='; '.join(reasons),
                         trace_status='F missing: mapping export only' if s.F is None else 'available'))
    return rows


def prepare_sessions(sessions, param):
    report = session_report(sessions, param)
    kept = [s for s, row in zip(sessions, report) if row['included']]
    if len(kept) < 2:
        raise ValueError('At least two included sessions are required. Check Session QC, '
                         'the minimum cell count, and session checkboxes.')
    if len({str(s.folder.resolve()) for s in kept}) != len(kept):
        raise ValueError('The same suite2p folder was added more than once.')
    if len({s.name for s in kept}) != len(kept):
        raise ValueError('Session names must be unique within a run.')
    if len({s.animal_id for s in kept if s.animal_id}) > 1:
        raise ValueError('Run one animal at a time; included sessions have different animal IDs.')
    if len({s.image_size for s in kept}) > 1:
        raise ValueError('Included sessions must have identical image dimensions.')
    planes = {s.folder.name for s in kept if s.folder.parent.name.lower() == 'suite2p'}
    if len(planes) > 1:
        raise ValueError('Run one imaging plane at a time.')
    if param.get('tracking_mode', 'global') == 'hub' and param.get('hub_ref') is not None:
        ref = int(param['hub_ref'])
        if ref < 0 or ref >= len(sessions) or not report[ref]['included']:
            raise ValueError('The selected Hub reference is excluded; choose an included session.')
    return kept, report
