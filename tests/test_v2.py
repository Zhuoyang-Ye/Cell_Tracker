import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.session import Session, _session_label
from core.quality import prepare_sessions, session_report
from core.tracking import (PairwiseCache, apply_tracking, compute_pairwise, run_batch,
                           one_to_one, _raw_pair, cache_is_valid)
from core.export import export_results, original_indices
from core.iou import compute_iou_matrix


def session(name, n=2, frames=4, folder=None):
    return Session(name, Path(folder or name),
        [np.array([2+i*3, 3+i*3]) for i in range(n)],
        [np.array([2, 2]) for i in range(n)],
        np.column_stack((np.ones(n), np.linspace(.99, .95, n))),
        np.zeros((20, 20), np.float32), (20, 20),
        F=np.arange(n*frames, dtype=np.float32).reshape(n, frames),
        Fneu=np.arange(n*frames, dtype=np.float32).reshape(n, frames)+100)


def make_cache(sizes, matrices):
    sessions = [session(f'A811-2026042{i+1}-01', size) for i, size in enumerate(sizes)]
    raw = {key: _raw_pair(np.array([0,0,0,1,1,0.]), np.asarray(m, dtype=float), 0)
           for key, m in matrices.items()}
    return PairwiseCache(sessions, [np.arange(n) for n in sizes],
                         [np.arange(n) for n in sizes], raw, ())


class NamingAndQuality(unittest.TestCase):
    def test_current_and_legacy_layouts(self):
        cases = {
            '/data/A811/20260422/01/suite2p/plane0': 'A811-20260422-01',
            '/data/A1234/20260422/2/suite2p/plane0': 'A1234-20260422-02',
            '/data/A752-20241031/02/suite2p/plane0': 'A752-20241031-02',
            '/data/A811-20260422-01/suite2p/plane0': 'A811-20260422-01',
            '/data/custom/suite2p/plane0': 'custom',
            '/data/unknown': 'unknown',
        }
        for path, expected in cases.items():
            self.assertEqual(_session_label(Path(path)), expected)

    def test_loader_validates_and_memory_maps(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'A811'/'20260422'/'01'/'suite2p'/'plane0'
            folder.mkdir(parents=True)
            s = session('fixture')
            np.save(folder/'ops.npy', dict(Ly=20, Lx=20, meanImg=s.mean_img))
            np.save(folder/'stat.npy', np.array([dict(xpix=x, ypix=y) for x,y in zip(s.xpix,s.ypix)], object))
            np.save(folder/'iscell.npy', s.iscell)
            np.save(folder/'F.npy', s.F)
            np.save(folder/'Fneu.npy', s.Fneu)
            loaded = Session.load(folder)
            self.assertEqual(loaded.name, 'A811-20260422-01')
            self.assertIsInstance(loaded.F, np.memmap)
            np.save(folder/'iscell.npy', np.ones((3,2)))
            with self.assertRaisesRegex(ValueError, 'changed'):
                loaded.validate()
            with self.assertRaisesRegex(ValueError, 'counts'):
                Session.load(folder)

    def test_shape_validation(self):
        s = session('A811-20260422-01')
        s.Fneu = np.zeros((2, 5))
        with self.assertRaisesRegex(ValueError, 'frame counts'):
            s.validate()
        s.Fneu = None
        s.xpix[0] = np.array([2.5, 3.])
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            s.validate()

    def test_quality_is_separate_from_tracking(self):
        s = session('A811-20260422-01', 3)
        s.iscell[:, 1] = [.95, .3, .1]
        report = session_report([s], dict(count_threshold=.9, tracking_prob_threshold=.2, min_cells_cutoff=2))
        self.assertEqual((report[0]['n_count'], report[0]['n_track']), (1,2))
        self.assertFalse(report[0]['included'])
        self.assertTrue(session_report([s], dict(min_cells_cutoff=0))[0]['included'])

    def test_mixed_animals_dimensions_and_exclusions(self):
        a, b = session('A811-20260422-01'), session('A812-20260422-01')
        with self.assertRaisesRegex(ValueError, 'one animal'):
            prepare_sessions([a,b], {})
        b.name = 'A811-20260423-01'
        b.image_size, b.mean_img = (21,20), np.zeros((21,20))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            prepare_sessions([a,b], {})
        with self.assertRaisesRegex(ValueError, 'At least two'):
            prepare_sessions([a,b], {'exclude_paths':[str(b.folder)]})

    def test_empty_tracking_is_reported(self):
        s = session('A811-20260422-01')
        s.iscell[:,1] = .1
        self.assertIn('no tracking ROIs', session_report([s], {})[0]['reason'])


class Matching(unittest.TestCase):
    def test_greedy_one_to_one_and_threshold(self):
        matrix = np.array([[.8, .7], [.75, .6]])
        self.assertEqual(one_to_one(matrix,.4).tolist(), [[0,0], [1,1]])
        self.assertEqual(one_to_one(matrix,.7).tolist(), [[0,0]])
        self.assertEqual(one_to_one(np.zeros((2,2)),0).shape, (0,2))
        self.assertEqual(one_to_one(np.empty((2,0)),.4).shape, (0,2))
        self.assertEqual(one_to_one(np.ones((2,2)),.4).tolist(), [[0,0],[1,1]])

    def test_collisions_never_duplicate_a_target(self):
        cache = make_cache([2,1], {(0,1): [[.8],[.7]]})
        for mode in ('chain','hub','global'):
            result = apply_tracking(cache, dict(tracking_mode=mode,hub_ref=0))
            self.assertEqual(result.pairs[0,1].collisions, 1)
            self.assertEqual(len(result.tracking_matrix), 2)
            for col in result.tracking_matrix.T:
                present = col[col >= 0]
                self.assertEqual(len(present), len(set(present)))

    def test_transitive_conflict_excluded_whole(self):
        cache = make_cache([2,1,1], {(0,1): [[.8],[.1]], (1,2): [[.8]], (0,2): [[.1],[.8]]})
        result = apply_tracking(cache, {})
        self.assertEqual(result.tracking_matrix.shape, (0,3))
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(len(result.conflicts[0]['members']), 4)
        self.assertEqual(result.presence_count.size, 0)

    def test_empty_groups_and_dilated_empty_iou(self):
        result = apply_tracking(make_cache([0,0], {}), {})
        self.assertEqual(result.tracking_matrix.shape, (0,2))
        out = compute_iou_matrix([np.array([1])], [np.array([1])], [], [], (10,10), 2)
        self.assertEqual(out.shape, (1,0))

    def test_hub_uses_canonical_pair_in_reverse(self):
        cache = make_cache([1,1], {(0,1): [[.8]]})
        result = apply_tracking(cache, dict(tracking_mode='hub', hub_ref=1))
        np.testing.assert_array_equal(result.tracking_matrix, [[0,0]])

    def test_chain_reserves_target_across_different_previous_sessions(self):
        cache = make_cache([1,1,1], {(0,1): [[0]], (0,2): [[.8]], (1,2): [[.9]]})
        result = apply_tracking(cache, dict(tracking_mode='chain'))
        self.assertEqual(int((result.tracking_matrix[:,2] >= 0).sum()), 1)
        self.assertTrue(any(np.array_equal(row, [-1,0,0]) for row in result.tracking_matrix))


class Caching(unittest.TestCase):
    def test_pairs_persist_subset_reuses_and_settings_invalidate(self):
        with tempfile.TemporaryDirectory() as folder:
            sessions = [session(f'A811-2026042{i+1}-01') for i in range(3)]
            param = dict(cache_dir=folder)
            with patch('core.tracking.find_alignment', return_value=np.array([0,0,0,1,1,0.])) as align:
                first = compute_pairwise(sessions, param)
                self.assertEqual(align.call_count, 3)  # not six directed registrations
                self.assertEqual(len(first.raw_pairs), 6)  # plus identities
                compute_pairwise(sessions, param)
                compute_pairwise(sessions[:2], param)
                self.assertEqual(align.call_count, 3)
                compute_pairwise(sessions, dict(param, tracking_prob_threshold=.96))
                self.assertEqual(align.call_count, 3)  # reuse transforms; recalculate IOU
                compute_pairwise(sessions, dict(param, mask_radius=1))
                self.assertEqual(align.call_count, 6)
                sessions[0].xpix[0] += 1
                compute_pairwise(sessions, param)
                self.assertEqual(align.call_count, 8)  # only two pairs with changed input
                self.assertFalse(cache_is_valid(first, sessions, param))

    def test_corrupt_cache_is_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            sessions = [session('A811-20260422-01'), session('A811-20260423-01')]
            with patch('core.tracking.find_alignment', return_value=np.array([0,0,0,1,1,0.])) as align:
                compute_pairwise(sessions, dict(cache_dir=folder))
                for path in Path(folder).glob('*.npz'):
                    path.write_bytes(b'broken')
                result = compute_pairwise(sessions, dict(cache_dir=folder))
                self.assertEqual(align.call_count, 2)
                self.assertTrue(result.cache_info['warnings'])

    def test_excluded_hub_reference_is_rejected(self):
        sessions = [session(f'A811-2026042{i+1}-01') for i in range(3)]
        with self.assertRaisesRegex(ValueError, 'reference is excluded'):
            run_batch(sessions, dict(tracking_mode='hub', hub_ref=0,
                                    exclude_paths=[str(sessions[0].folder)]))

    def test_hub_reference_remapped_after_exclusion(self):
        sessions = [session(f'A811-2026042{i+1}-01') for i in range(3)]
        with patch('core.tracking.find_alignment', return_value=np.array([0,0,0,1,1,0.])):
            r = run_batch(sessions, dict(tracking_mode='hub', hub_ref=2,
                          exclude_paths=[str(sessions[0].folder)]))
        self.assertEqual(r.param['hub_ref'], 1)
        self.assertEqual(len(r.session_qc), 3)
        self.assertFalse(r.session_qc[0]['included'])


class Export(unittest.TestCase):
    def result(self):
        cache = make_cache([2,1], {(0,1): [[.8],[0.]]})
        cache.sessions[1].F = np.array([[10,20,30]], dtype=np.float32)
        cache.sessions[1].Fneu = None
        return apply_tracking(cache, dict(max_missing_sessions=1))

    def test_traces_nan_padding_matlab_indexing_and_manifest(self):
        r = self.result()
        with tempfile.TemporaryDirectory() as root:
            out = export_results(r, root, matlab=True)
            np.testing.assert_array_equal(np.load(out/'unified_original_roi_indices.npy'), [[0,0],[1,-1]])
            trace = np.load(out/f'unified_F_{r.sessions[1].name}.npy')
            self.assertEqual(trace.shape, (2,3))
            np.testing.assert_array_equal(trace[0], [10,20,30])
            self.assertTrue(np.isnan(trace[1]).all())
            self.assertTrue((out/f'unified_Fneu_{r.sessions[0].name}.npy').exists())
            ids = loadmat(out/f'{r.sessions[1].name}_cellidx.mat')['cellID'][:,0]
            self.assertEqual(ids[0], 1)
            self.assertTrue(np.isnan(ids[1]))
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['n_unified'], 2)
            self.assertEqual(len(manifest['sessions']), 2)
            self.assertTrue(np.isnan(np.load(out/'v_opt_all.npy')[1,0]).all())
            second = export_results(r, root, include_traces=False)
            self.assertNotEqual(out, second)
            self.assertFalse(list(second.glob('unified_F_*.npy')))

    def test_export_filter_does_not_change_diagnostics(self):
        r = self.result()
        r.param['max_missing_sessions'] = 0
        self.assertEqual(len(r.tracking_matrix), 2)
        self.assertEqual(r.export_rows.tolist(), [0])
        with tempfile.TemporaryDirectory() as root:
            out = export_results(r, root)
            self.assertEqual(np.load(out/'unified_original_roi_indices.npy').shape, (1,2))

    def test_missing_F_no_partial_export(self):
        r = self.result()
        r.sessions[1].F = None
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'missing'):
                export_results(r, root)
            self.assertEqual(list(Path(root).iterdir()), [])
            export_results(r, root, include_traces=False)

    def test_failed_write_cleans_staging(self):
        r = self.result()
        with tempfile.TemporaryDirectory() as root:
            with patch('core.export._write_trace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    export_results(r, root)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_zero_surviving_tracks_exports_valid_empty_arrays(self):
        r = apply_tracking(make_cache([2,1,1], {(0,1):[[.8],[.1]], (1,2):[[.8]], (0,2):[[.1],[.8]]}), {})
        with tempfile.TemporaryDirectory() as root:
            out = export_results(r, root, matlab=True)
            self.assertEqual(np.load(out/f'unified_F_{r.sessions[0].name}.npy').shape, (0,4))


if __name__ == '__main__':
    unittest.main()
