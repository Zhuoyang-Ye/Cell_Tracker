"""Consistent-row trace export with explicit Python/MATLAB index conventions."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import re
import shutil
import tempfile
import numpy as np


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def original_indices(result):
    mapping = np.full(result.tracking_matrix.shape, -1, dtype=np.int32)
    for s in range(len(result.sessions)):
        present = result.tracking_matrix[:, s] >= 0
        mapping[present, s] = result.track_idx[s][result.tracking_matrix[present, s]]
    return mapping


def _write_trace(path, trace, ids):
    """Write one session at a time, in bounded chunks; absent cells remain NaN."""
    dtype = np.result_type(trace.dtype, np.float32)
    out = np.lib.format.open_memmap(path, mode='w+', dtype=dtype,
                                  shape=(len(ids), trace.shape[1]))
    out[:] = np.nan
    for start in range(0, len(ids), 32):
        rows = np.arange(start, min(start + 32, len(ids)))
        present = rows[ids[rows] >= 0]
        out[present] = trace[ids[present]]
    out.flush()
    del out


def export_results(result, output_parent, include_traces=True, matlab=False):
    """Publish a fresh subfolder only after the whole export succeeds."""
    if include_traces:
        for session in result.sessions:
            session.validate()
            if session.F is None:
                raise ValueError(f'{session.name}: F.npy is missing. Choose mapping-only export.')
    names = [s.name for s in result.sessions]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', n) for n in names):
        raise ValueError('Export requires unique session names containing letters, numbers, ., _, or -.')
    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stage = Path(tempfile.mkdtemp(prefix=f'.roi_tracking_{stamp}_', dir=parent))
    destination = parent / stage.name.lstrip('.')
    try:
        _write_export(result, stage, include_traces, matlab)
        stage.rename(destination)
    except Exception:
        shutil.rmtree(stage)
        raise
    return destination


def _write_export(r, out, include_traces, matlab):
    mapping = original_indices(r)
    rows = r.export_rows
    unified = mapping[rows]
    np.save(out / 'tracking_matrix.npy', r.tracking_matrix)
    np.save(out / 'presence_count.npy', r.presence_count)
    np.save(out / 'original_roi_indices.npy', mapping)
    np.save(out / 'unified_track_rows.npy', rows)
    np.save(out / 'unified_original_roi_indices.npy', unified)
    np.save(out / 'shared_cells.npy', r.shared_cells)
    n = len(r.sessions)
    transforms = np.full((n, n, 6), np.nan)
    pair_rows = []
    for (i, j), pr in r.pairs.items():
        transforms[i, j] = pr.v_opt
        if i != j:
            pair_rows.append(dict(source=r.sessions[i].name, target=r.sessions[j].name,
                                  matched=len(pr.matched_a), collisions=pr.collisions,
                                  overlap_px=pr.overlap_px))
    np.save(out / 'v_opt_all.npy', transforms)
    for s, session in enumerate(r.sessions):
        np.save(out / f'tracking_orig_idx_{session.name}.npy', mapping[:, s])
        if include_traces:
            for key, trace in (('F', session.F), ('Fneu', session.Fneu)):
                if trace is not None:
                    _write_trace(out / f'unified_{key}_{session.name}.npy', trace, unified[:, s])
        if matlab:
            from scipy.io import savemat
            ids = unified[:, s].astype(float) + 1
            ids[unified[:, s] < 0] = np.nan
            savemat(out / f'{session.name}_cellidx.mat', {'cellID': ids[:, None]})
            if include_traces:
                traces = {'F': np.load(out / f'unified_F_{session.name}.npy', mmap_mode='r')}
                if session.Fneu is not None:
                    traces['Fneu'] = np.load(out / f'unified_Fneu_{session.name}.npy', mmap_mode='r')
                savemat(out / f'{session.name}_Fu.mat', traces)
    if matlab:
        from scipy.io import savemat
        rowmap = unified.astype(float) + 1
        rowmap[unified < 0] = np.nan
        savemat(out / 'unified_meta.mat', dict(rowmap=rowmap,
                session_names=np.array([s.name for s in r.sessions], dtype=object),
                row_presence=r.presence_count[rows][:, None],
                tracking_rows_matlab=(rows + 1)[:, None]))
    manifest = dict(format_version=2, sessions=[dict(name=s.name, folder=str(s.folder),
                    animal=s.animal_id, image_size=s.image_size,
                    F_shape=None if s.F is None else s.F.shape,
                    Fneu_shape=None if s.Fneu is None else s.Fneu.shape,
                    source_stamp=s.loaded_stamp) for s in r.sessions],
                    params=r.param, session_qc=r.session_qc, conflicts=r.conflicts,
                    cache_info=r.cache_info, n_unified=len(rows),
                    minimum_presence=max(1, n-r.param.get('max_missing_sessions', 2)),
                    traces_exported=include_traces,
                    conventions=dict(tracking_matrix='0-based local index in per-session track_idx; -1 absent',
                        original_roi_indices='0-based row in suite2p stat/F/Fneu; -1 absent',
                        unified_track_rows='0-based rows of tracking_matrix selected for all unified outputs',
                        matlab_cellID='1-based original suite2p ROI row; NaN absent',
                        traces='Raw F and Fneu, uncorrected; NaN rows for absent ROIs; session durations may differ',
                        v_opt_all='Only i<=j computed, direction i->j; lower triangle NaN, not inverse transforms',
                        shared_cells='One-to-one pair counts before global conflict rejection',
                        conflicts='Excluded from tracking_matrix and every unified export; member indices are 0-based'))
    for filename, value in (('manifest.json', manifest), ('param.json', r.param), ('conflicts.json', r.conflicts)):
        (out / filename).write_text(json.dumps(value, indent=2, default=_json_default))
    for filename, records in (('session_qc.csv', r.session_qc), ('pair_summary.csv', pair_rows)):
        if records:
            with (out / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    for s, indices in enumerate(r.track_idx):
        np.save(out / f'track_idx_{r.sessions[s].name}.npy', indices)
    # Recheck all inputs before publishing files, including mapping-only exports.
    for session in r.sessions:
        session.validate()
