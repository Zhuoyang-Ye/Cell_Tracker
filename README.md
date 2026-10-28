# ROI Alignment & Tracking — Python V2

The primary dataset for this application is GACh–GECO two-photon imaging.

Run `python main.py` from this folder in the `analysis_mpfi` environment.
The GUI uses the parent pipeline's `Utilities.paper_style` for diagnostic figures.

## Workflow

1. Add suite2p plane folders. Both `A811/20260422/01/suite2p/plane0` and the
   older `A752-20241031/02/suite2p/plane0` naming layouts are recognised.
2. Inspect **Session QC**. Uncheck sessions to exclude them. Each run uses one
   animal, one plane and one image size. Session order is the order loaded
   (also used by Chain mode); no automatic chronological reordering is done.
3. Choose a parameter preset and click **Apply**, or edit individual controls.
4. Run alignment and tracking. Inspect Tracking, Shared cells, Presence,
   IOU and Stats / conflicts. Changing analysis settings clears old results;
   run again to rebuild them, reusing valid cached work.
5. Set **Max missing sessions** and run again to update the export selection.
   Zero requires presence in every included session. With `n` included sessions
   and value `k`, export requires presence in at least `max(1, n-k)` sessions.
6. Export to a new results subfolder. Unified raw F and available Fneu are
   optional; MATLAB files can also be selected. Missing F disables trace export,
   but mapping export remains available. Fneu is exported only where available.

The ROI canvas's existing double-click classification toggle is a display preview;
it does not edit suite2p files or change probability-based tracking selection.

## What the preset values mean

A default is simply a value prefilled when the application opens. The application
starts with **MATLAB V2 settings**. The previous Python numeric settings are still
available as a preset; both presets use the corrected matching algorithms.

| Control | MATLAB V2 | Previous Python | Meaning |
|---|---:|---:|---|
| Align top-N | 200 | 300 | Highest-probability ROIs used to find the transform |
| Track probability ≥ | 0.2 | 0 | ROIs included in matching; zero includes all |
| Match IOU ≥ | 0.4 | 0.3 | Required mask intersection-over-union |
| Search step scale | 0.5 | 1.0 | Optimizer step-size scaling |
| Max evaluations / start | 2000 | 5000 | Evaluation limit for each of five starts |
| Quality probability ≥ | 0.9 | 0.9 | Separate threshold for session quality counts |
| Min high-confidence ROIs | 60 | 0 | Exclude sessions below this count; zero disables |
| Max missing sessions | 2 | 2 | Unified-export selection; newly added to Python |

The V2 minimum count of 60 is inherited from the MATLAB settings. Inspect the
GACh–GECO session counts in Session QC to decide whether this cutoff suits the
sessions being analysed; set it to zero to disable the cutoff. Changing quality
thresholds does not change the top-N alignment subset or the tracking probability
threshold.

## Matching and registration

* Each unordered pair is registered once, in loaded order (`i -> j`, `i < j`).
  The same overlap matrix supports matching in both directions. Hub uses the
  transposed correspondence when its reference is the later session; it does
  not run a second reverse registration.
* Candidate matches are processed in descending IOU, reserving each ROI once.
  Ties use MATLAB column-major order. Zero overlap never matches, even with
  IOU threshold zero. This is V2's greedy matching, not an optimal assignment solver.
* Global mode groups all one-to-one matches. A group containing multiple ROIs
  from any single session is ambiguous: the whole group is excluded and its
  original ROI IDs are listed in Stats / conflicts and `conflicts.json`.
* Chain and Hub also enforce unique target use. Chain can bridge missing
  sessions using the last observation; Hub keeps nonreference unmatched ROIs
  as standalone tracks. Global remains the default for V2 processing.
* Presence statistics include all unambiguous tracks, before the export
  presence cutoff. Shared-cell counts describe pair matches before global
  conflict rejection. These are different stages and need not equal export counts.
* Rows are ordered by decreasing presence, then decreasing mean iscell
  probability, with a deterministic tie-break. Track row numbers are local
  to a run and should not be treated as permanent IDs across parameter changes.

## Cache and validation

`python/.alignment_cache/` stores versioned numeric NPZ files. Disable disk reuse
with the checkbox; the current run's in-memory cache is still available.
Registration and IOU have separate cache keys, so changing tracking probability
can reuse registration, while changing IOU threshold or export selection needs
only rematching. Adding/removing sessions reuses unaffected disk-cached pairs.
Mask radius, optimizer settings, image size, ROI coordinates/probabilities and
input identities participate in invalidation. Changed suite2p files require
reloading the session. A broken cache is ignored and recomputed; delete the
cache directory at any time when the application is idle.

The loader checks coordinate validity, stat/iscell/F/Fneu row counts, image
size and trace frame consistency. Traces are memory mapped rather than loaded
in full. Sessions with no tracking ROIs are excluded before registration.

## Export conventions

Each successful export creates a fresh `roi_tracking_<UTC timestamp>_<id>`
subfolder. Source suite2p files are never overwritten. Failed exports remove
their staging folder.

* `tracking_matrix.npy`: all unambiguous tracks × included sessions; values are
  **0-based local indices** into each saved `track_idx_<session>.npy`; `-1` absent.
* `original_roi_indices.npy` and `tracking_orig_idx_<session>.npy`: corresponding
  **0-based original suite2p row indices**; `-1` absent.
* `unified_track_rows.npy`: 0-based tracking-matrix rows selected by the presence
  cutoff. `unified_original_roi_indices.npy` has those same rows and session order.
* `unified_F_<session>.npy`, `unified_Fneu_<session>.npy`: raw traces in the
  unified row order, **NaN rows for absent cells**. Frame counts may differ
  between sessions. No neuropil correction, normalization or dFF is performed.
* Optional `<session>_Fu.mat`: variables `F` and, where available, `Fneu`.
  `<session>_cellidx.mat` contains `cellID`: **1-based original suite2p ROI indices**,
  NaN absent. `unified_meta.mat` contains the full 1-based rowmap and session names.
* `v_opt_all.npy`: six-parameter transforms for `i <= j`; **lower triangle is NaN**
  because reverse registrations were not calculated. Do not interpret these
  entries as zeros or inverse transforms.
* `manifest.json`: ordered session names/paths, input stamps, parameters, index
  conventions, trace availability, QC, conflicts and cache diagnostics.
  `param.json`, `session_qc.csv`, `pair_summary.csv`, `shared_cells.npy`,
  `presence_count.npy` and `conflicts.json` provide convenient individual outputs.

Both the new matching rules and parameter presets can change results relative to
previous Python exports. Validate representative biological sessions before
combining results from the old and new versions.

## Verification

Run core and offscreen GUI tests from this folder with:

```sh
python -B -m unittest discover -s tests -v
```
