from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from .alignment import find_alignment, apply_alignment
from .roi_mask import create_roi_mask
from .iou import compute_iou_matrix
from .quality import prepare_sessions
from .cache import PairStore, fingerprint, alignment_settings, cache_key


@dataclass
class PairResult:
    i: int
    j: int
    v_opt: np.ndarray
    iou_matrix: np.ndarray
    max_iou: np.ndarray
    best_match: np.ndarray
    matched_a: np.ndarray
    matched_b: np.ndarray
    overlap_px: int
    collisions: int = 0


@dataclass
class BatchResult:
    sessions: list
    align_idx: list
    track_idx: list
    pairs: dict  # i <= j only; registration is i -> j
    tracking_matrix: np.ndarray  # local tracking indices, -1 = absent
    presence_count: np.ndarray
    param: dict
    conflicts: list = field(default_factory=list)
    session_qc: list = field(default_factory=list)
    shared_cells: Optional[np.ndarray] = None
    cache_info: dict = field(default_factory=dict)

    @property
    def export_rows(self):
        minimum = max(1, len(self.sessions) - self.param.get('max_missing_sessions', 2))
        return np.flatnonzero(self.presence_count >= minimum)


@dataclass
class PairwiseCache:
    sessions: list
    align_idx: list
    track_idx: list
    raw_pairs: dict
    param_key: tuple
    cache_info: dict = field(default_factory=dict)


def one_to_one(iou, threshold):
    """V2 greedy assignment. Ties use MATLAB's column-major candidate order.

    Zero overlap is never a match, even when the threshold is set to zero.
    """
    iou = np.asarray(iou)
    flat = iou.ravel(order='F')
    candidates = np.flatnonzero(np.isfinite(flat) & (flat > 0) & (flat >= threshold))
    order = candidates[np.argsort(-flat[candidates], kind='stable')]
    used_a, used_b, pairs = set(), set(), []
    for index in order:
        a, b = int(index % iou.shape[0]), int(index // iou.shape[0])
        if a not in used_a and b not in used_b:
            used_a.add(a)
            used_b.add(b)
            pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def _directed_matches(pairs, source, target):
    """Reuse one registration frame in either matching direction, without inventing a transform."""
    key = (min(source, target), max(source, target))
    pr = pairs.get(key)
    if pr is None:
        return []
    matches = zip(pr.matched_a, pr.matched_b)
    if source <= target:
        return [(int(a), int(b), float(pr.iou_matrix[a, b])) for a, b in matches]
    return [(int(b), int(a), float(pr.iou_matrix[a, b])) for a, b in matches]


def _build_tracking_chain(n, track_idx, pairs, iou_thr):
    tracked = []
    for t in range(n):
        candidates = []
        for row_idx, row in enumerate(tracked):
            previous = np.flatnonzero(row[:t] >= 0)
            if not len(previous):
                continue
            prev = int(previous[-1])
            for a, b, score in _directed_matches(pairs, prev, t):
                if a == row[prev]:
                    candidates.append((-score, row_idx, b))
        # Different previous sessions can propose the same target. Reserve it once.
        used_rows, used_targets = set(), set()
        for _, row_idx, target in sorted(candidates):
            if row_idx not in used_rows and target not in used_targets:
                tracked[row_idx][t] = target
                used_rows.add(row_idx)
                used_targets.add(target)
        for local in range(len(track_idx[t])):
            if local not in used_targets:
                row = np.full(n, -1, dtype=np.int32)
                row[t] = local
                tracked.append(row)
    return tracked


def _build_tracking_hub(n, track_idx, pairs, iou_thr, hub_ref):
    ref = n // 2 if hub_ref is None else int(hub_ref)
    if not 0 <= ref < n:
        raise ValueError('Hub reference is outside the included session list.')
    tracked = [np.full(n, -1, dtype=np.int32) for _ in track_idx[ref]]
    for local, row in enumerate(tracked):
        row[ref] = local
    for t in range(n):
        if t == ref:
            continue
        used = set()
        for a, b, score in _directed_matches(pairs, ref, t):
            tracked[a][t] = b
            used.add(b)
        for local in range(len(track_idx[t])):
            if local not in used:
                row = np.full(n, -1, dtype=np.int32)
                row[t] = local
                tracked.append(row)
    return tracked


def _global_components(n, track_idx, pairs):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    sizes = [len(t) for t in track_idx]
    offsets = np.r_[0, np.cumsum(sizes)]
    total = int(offsets[-1])
    src, dst = [], []
    for (i, j), pr in pairs.items():
        if i >= j:
            continue
        src.extend((offsets[i] + pr.matched_a).tolist())
        dst.extend((offsets[j] + pr.matched_b).tolist())
    graph = coo_matrix((np.ones(len(src), dtype=bool), (src, dst)), shape=(total, total))
    _, labels = connected_components(graph.tocsr(), directed=False)
    components = {}
    for s, size in enumerate(sizes):
        for local in range(size):
            comp = int(labels[offsets[s] + local])
            components.setdefault(comp, []).append((s, local))
    tracked, conflicts = [], []
    for comp, members in components.items():
        sessions = [s for s, _ in members]
        if len(set(sessions)) != len(sessions):
            conflicts.append(dict(component=comp, presence=len(set(sessions)), members=[
                dict(session=s, local_roi=local, original_roi=int(track_idx[s][local]))
                for s, local in members]))
            continue
        row = np.full(n, -1, dtype=np.int32)
        for s, local in members:
            row[s] = local
        tracked.append(row)
    return tracked, conflicts


def _build_tracking_global(n, track_idx, pairs, iou_thr):
    return _global_components(n, track_idx, pairs)[0]


def _pairwise_param_key(sessions, param):
    return (tuple(fingerprint(s) for s in sessions),
            cache_key(alignment_settings(param), param.get('tracking_prob_threshold', .2)))


def cache_is_valid(cache, sessions, param):
    if cache is None:
        return False
    kept, _ = prepare_sessions(sessions, param)
    return cache.param_key == _pairwise_param_key(kept, param)


def _raw_pair(v_opt, matrix, overlap):
    na, nb = matrix.shape
    return dict(v_opt=v_opt, iou_matrix=matrix,
                max_iou=matrix.max(axis=1) if nb else np.zeros(na),
                best_match=matrix.argmax(axis=1) if nb else np.full(na, -1, dtype=int),
                overlap_px=int(overlap))


def compute_pairwise(sessions, param, pair_progress_cb=None, step_progress_cb=None):
    """Compute each unordered pair once; persist registration and IOU separately."""
    sessions, _ = prepare_sessions(sessions, param)
    n = len(sessions)
    align_idx, track_idx = [], []
    for s in sessions:
        align_idx.append(np.argsort(-s.iscell_prob, kind='stable')[:param.get('alignment_top_n', 200)])
        track_idx.append(np.flatnonzero(s.iscell_prob >= param.get('tracking_prob_threshold', .2)))
    store = PairStore(param.get('cache_dir') if param.get('use_disk_cache', True) else None)
    ids = [fingerprint(s) for s in sessions]
    raw_pairs = {}
    pair_num, computed = 0, 0
    for i in range(n):
        nt = len(track_idx[i])
        raw_pairs[i, i] = _raw_pair(np.array([0, 0, 0, 1, 1, 0.]), np.eye(nt), 0)
        for j in range(i + 1, n):
            pair_num += 1
            if pair_progress_cb:
                pair_progress_cb(pair_num, n * (n - 1) // 2,
                                 f'{sessions[i].name} → {sessions[j].name}')
            si, sj = sessions[i], sessions[j]
            ai, aj, ti, tj = align_idx[i], align_idx[j], track_idx[i], track_idx[j]
            akey = cache_key('alignment', ids[i], ids[j], alignment_settings(param))
            ikey = cache_key('iou', akey, param.get('tracking_prob_threshold', .2))
            saved = store.load(ikey)
            if (saved is not None and {'v_opt', 'iou_matrix', 'overlap_px'} <= saved.keys()
                    and saved['iou_matrix'].shape == (len(ti), len(tj))
                    and saved['v_opt'].shape == (6,)
                    and saved['overlap_px'].shape == ()
                    and np.isfinite(saved['overlap_px']) and saved['overlap_px'] >= 0
                    and np.all(np.isfinite(saved['v_opt']))
                    and np.all(np.isfinite(saved['iou_matrix']))
                    and np.all((saved['iou_matrix'] >= 0) & (saved['iou_matrix'] <= 1))):
                raw_pairs[i, j] = _raw_pair(saved['v_opt'], saved['iou_matrix'], saved['overlap_px'])
                continue
            saved = store.load(akey)
            if (saved is not None and 'v_opt' in saved and saved['v_opt'].shape == (6,)
                    and np.all(np.isfinite(saved['v_opt']))):
                v_opt = saved['v_opt']
            else:
                v_opt = find_alignment(
                    [si.xpix[k] for k in ai], [si.ypix[k] for k in ai],
                    [sj.xpix[k] for k in aj], [sj.ypix[k] for k in aj], si.image_size, param,
                    progress_callback=lambda k, total: step_progress_cb('alignment', k, total)
                    if step_progress_cb else None)
                computed += 1
                store.save(akey, v_opt=v_opt)
            xa, ya = apply_alignment([si.xpix[k] for k in ai], [si.ypix[k] for k in ai], v_opt, si.image_size)
            radius = param.get('mask_radius', 0)
            ma = create_roi_mask(xa, ya, si.image_size, radius)
            mb = create_roi_mask([sj.xpix[k] for k in aj], [sj.ypix[k] for k in aj], sj.image_size, radius)
            overlap = int(np.count_nonzero(ma & mb))
            xa, ya = apply_alignment([si.xpix[k] for k in ti], [si.ypix[k] for k in ti], v_opt, si.image_size)
            matrix = compute_iou_matrix(
                xa, ya, [sj.xpix[k] for k in tj], [sj.ypix[k] for k in tj], si.image_size, radius,
                progress_callback=lambda k, total: step_progress_cb('iou', k, total)
                if step_progress_cb else None)
            raw_pairs[i, j] = _raw_pair(v_opt, matrix, overlap)
            store.save(ikey, v_opt=v_opt, iou_matrix=matrix, overlap_px=overlap)
    # Detect source changes during a long run before returning any result.
    for s in sessions:
        s.validate()
    return PairwiseCache(sessions, align_idx, track_idx, raw_pairs,
                         _pairwise_param_key(sessions, param),
                         dict(registrations_computed=computed, disk_hits=store.hits, warnings=store.warnings))


def apply_tracking(cache, param):
    n = len(cache.sessions)
    thr = param.get('iou_threshold', .4)
    if not 0 <= thr <= 1:
        raise ValueError('IOU threshold must be between 0 and 1.')
    if param.get('max_missing_sessions', 2) < 0:
        raise ValueError('Maximum missing sessions must be nonnegative.')
    pairs = {}
    shared = np.diag([len(t) for t in cache.track_idx])
    for (i, j), raw in cache.raw_pairs.items():
        matches = one_to_one(raw['iou_matrix'], thr)
        above = (raw['max_iou'] >= thr) & (raw['max_iou'] > 0)
        best = raw['best_match'][above]
        pairs[i, j] = PairResult(i=i, j=j, **raw, matched_a=matches[:, 0], matched_b=matches[:, 1],
                                collisions=int(len(best) - len(np.unique(best))))
        shared[i, j] = shared[j, i] = len(matches)
    mode = param.get('tracking_mode', 'global')
    conflicts = []
    if mode == 'global':
        tracked, conflicts = _global_components(n, cache.track_idx, pairs)
    elif mode == 'chain':
        tracked = _build_tracking_chain(n, cache.track_idx, pairs, thr)
    elif mode == 'hub':
        tracked = _build_tracking_hub(n, cache.track_idx, pairs, thr, param.get('hub_ref'))
    else:
        raise ValueError(f'Unknown tracking mode: {mode}')
    matrix = np.asarray(tracked, dtype=np.int32).reshape(-1, n)
    presence = (matrix >= 0).sum(axis=1)
    means = []
    for row in matrix:
        means.append(np.mean([cache.sessions[s].iscell_prob[cache.track_idx[s][local]]
                              for s, local in enumerate(row) if local >= 0]))
    order = np.lexsort((np.arange(len(matrix)), -np.asarray(means), -presence))
    return BatchResult(cache.sessions, cache.align_idx, cache.track_idx, pairs,
                       matrix[order], presence[order], dict(param), conflicts=conflicts,
                       shared_cells=shared, cache_info=dict(cache.cache_info))


def run_batch(sessions, param, pair_progress_cb=None, step_progress_cb=None):
    kept, report = prepare_sessions(sessions, param)
    run_param = dict(param)
    if param.get('tracking_mode') == 'hub' and param.get('hub_ref') is not None:
        reference = sessions[int(param['hub_ref'])]
        run_param['hub_ref'] = next(i for i, s in enumerate(kept) if s is reference)
    cache = compute_pairwise(kept, run_param, pair_progress_cb, step_progress_cb)
    result = apply_tracking(cache, run_param)
    result.session_qc = report
    return result
