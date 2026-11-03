# vicon2mano — API reference for agents

Machine-assisted index of every public function/class in the repo (tests excluded).
Read `CLAUDE.md` first for project state and caveats; per-strategy detail is in
`docs/strategies/*.md`. This file answers "which function does X, and where?".

## How to use this file

- Python env: `.viconvenv/Scripts/python.exe`. Tests: `.viconvenv/Scripts/python.exe -m pytest tests/ -q`.
- Entry points: CLI `vicon2mano <input> <output>` (`vicon2mano/core/cli.py:main`); everything else is a script in `scripts/<strategy>/` run as `python scripts/<strategy>/<name>.py --help`.
- Locate a trial with `core.dataset.find_trial_csv`; never import a strategy driver just for plumbing.
- Anything used by more than one strategy lives in `vicon2mano/core/`. `strategies/soma` imports `core` only.
- Always convert smplx output with `core.fitter.mano_output_to_joints21` (never `out.joints[:, :21]`).
- Marker→joint assignment goes through `core.correspondence.get_assignment` (label seed → DeepLabeler → Hungarian).
- Treat Vicon labels as unreliable (see "State of the data" in `CLAUDE.md`); validate repairs against bone-length checks (`core.bones`) or the human-verified `P10/Trial2_handsonly_manuallylabelled_filled.csv`.

## Map of the code

| Area | Purpose |
|---|---|
| `vicon2mano/core/` | loader, export, MANO fitter, correspondence, bones/continuity/agreement/quality checks, dataset lookup, viz palette |
| `strategies/cascade` | geometric correct/incorrect/missing classification, chain of trust from forearm |
| `strategies/mano` | MANO-model-based relabelling |
| `strategies/gmm` | spatial GMM + Kalman + Viterbi labeller |
| `strategies/deep` | 3-D U-Net labeller and synthetic/real data builders |
| `strategies/template` | distance-descriptor match to a labelled template |
| `strategies/soma` | SOMA-style learned labeller (library, synth, model, realdata, calibrate, infer) |
| `scripts/<strategy>/` | runnable drivers, reports, validation, visualisation |

Signatures below are auto-extracted; private (`_name`) helpers are omitted. Regenerate with
`.viconvenv/Scripts/python.exe scripts/shared/gen_api_reference.py` (only the part above the marker line is hand-editable).

<!-- GENERATED BELOW: do not edit -->

## Package: vicon2mano/core

### `vicon2mano/core/__init__.py`

Strategy-agnostic building blocks.


### `vicon2mano/core/agreement.py`

Frame-by-frame agreement between two labellings of the same recording.

- `base_name(label: str) -> str` â€” Strip a Vicon subject prefix: "P10:Palm1" -> "Palm1".
- **class `MarkerMatch`** â€” (no docstring)
- `match_markers(labels_a: list[str], labels_b: list[str]) -> MarkerMatch` â€” Pair up two marker-column lists by base name (prefix stripped).
- `per_frame_table(markers_a: np.ndarray, markers_b: np.ndarray, match: MarkerMatch, *, label_a: str='vicon', label_b: str='manual') -> pd.DataFrame` â€” Long-format per-frame, per-marker comparison for the common markers.
- `add_threshold_columns(df: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame` â€” Add ``within_<t>mm`` boolean columns for each configured threshold.
- `flag_temporal_jumps(df: pd.DataFrame, *, k: float=6.0, min_jump_mm: float=2.0) -> pd.DataFrame` â€” Flag frames whose error jumps far from its own marker's typical level.
- `temporal_events(df: pd.DataFrame) -> pd.DataFrame` â€” Collapse ``error_jump_flag`` rows into contiguous per-marker events.
- `missingness_events(df: pd.DataFrame) -> pd.DataFrame` â€” Contiguous per-marker runs of missing coordinates (trajectory gaps).
- `per_marker_event_summary(events: pd.DataFrame) -> pd.DataFrame` â€” One row per marker: event count, durations, and type breakdown.
- `per_marker_summary(df: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame` â€” One row per marker: match counts, missingness, and error stats.
- `overall_summary(df: pd.DataFrame, thresholds_mm: list[float]) -> dict` â€” Single-row dict of trial-wide agreement stats (see ``per_marker_summary`` for the same statistics broken out per marker).
- `reliability_table(marker_summary: pd.DataFrame, event_summary: pd.DataFrame, missingness: pd.DataFrame) -> pd.DataFrame` â€” One row per marker, combining availability, identity, and event burden into the three separately-meaningful quantities a reader actually needs:
- `cross_trial_summary(all_reliability: pd.DataFrame) -> pd.DataFrame` â€” One row per marker: how its :func:`reliability_table` numbers vary across several trials (``all_reliability`` = that table from each trial, concatenated, tagged by whatever trial-identifying columns the caller used -- only ``marker`` and the reliability columns are read here).

### `vicon2mano/core/bones.py`

Bone-length self-consistency: a model-free way to score whether a frame's marker labelling is geometrically plausible.

- `modal_bone_lengths(markers: np.ndarray, bones: list[tuple[int, int]], *, bin_mm: float=0.5, refine_mm: float=2.0) -> np.ndarray` â€” Reference length per bone from each bone's *marginal* mode.
- `consensus_bone_lengths(markers: np.ndarray, bones: list[tuple[int, int]], *, tol_mm: float=4.0, n_hypotheses: int=400, n_compare: int=3000, seed: int=0, min_...)` â€” Reference lengths from the largest mutually-consistent set of frames.
- `bone_error(markers: np.ndarray, bones: list[tuple[int, int]], ref: np.ndarray) -> np.ndarray` â€” (T, n_bones) absolute deviation from each bone's reference length.

### `vicon2mano/core/cli.py`

Command-line entry point: vicon2mano <input> <output> [options]

- `main(argv: list[str] | None=None) -> None` â€” (no docstring)

### `vicon2mano/core/continuity.py`

Trajectory continuity: does a marker move smoothly frame-to-frame, or does it jump -- a signal distinct from, and never combined with, geometric self-consistency (bone-length plausibility) or anchor-frame validity. A marker can have a perfectly plausible bone length while still having teleported ...

- `flag_discontinuities(markers: np.ndarray, *, k: float=6.0, min_jump_mm: float=25.0) -> np.ndarray` â€” (T, N) bool: frame t is flagged for marker n if the frame-to-frame displacement arriving at t is an outlier for that marker.

### `vicon2mano/core/correspondence.py`

Infer soft correspondences between Vicon markers and MANO joints.

- `label_seed(marker_labels: list[str], *, side: str | None=None) -> dict[str, int] | None` â€” Try to seed correspondences from marker label strings.
- `hungarian_assignment(markers: np.ndarray, mano_joints: np.ndarray, *, z_thresh: float=2.5) -> np.ndarray` â€” Return index array `assign` of length J.
- `sequence_assignment(markers_seq: np.ndarray, mano_joints_seq: np.ndarray, *, per_frame: bool=False) -> np.ndarray` â€” Compute marker↔joint assignment for a whole sequence.
- `get_assignment(markers_seq: np.ndarray, mano_joints_seq: np.ndarray, *, labeler=None, per_frame: bool=False, marker_labels: list[str] | None=None, side: str ...)` â€” Return a marker↔joint assignment, trying strategies in priority order.
- `relabel_by_template(markers: np.ndarray, template_markers: np.ndarray, template_labels: list[str], *, stride: int=20, normalize: bool=True) -> tuple[list[str...)` â€” Assign template labels to an unlabeled marker cloud by geometry only.
- `relabel_two_hands(markers: np.ndarray, template_markers: np.ndarray, template_labels: list[str], *, stride: int=20) -> tuple[list[str], np.ndarray]` â€” Relabel a complete two-hand cloud, chirality-safe.

### `vicon2mano/core/dataset.py`

Locating recordings on disk: participants, trials, statics, references.

- `load_ref_ranges_csv(path: str, participant: str, trial: str) -> np.ndarray` â€” Look up manually-verified good frames from the manual-frames log.
- `participant_sort_key(name: str) -> tuple[int, str]` â€” Sort "P2" before "P10" — plain string order puts P10 first, which makes the all-participants summary grid read wrong.
- `resolve_participants(requested: list[str], manual_sessions: dict[str, dict[str, str]], source: str='manual_frames.csv') -> list[str]` â€” Expand ["all"] to every participant with rows in manual_frames.csv, in numeric order. Any other list is returned as given (order preserved), with a warning for names the log doesn't know about — those would otherwise just silently produce an empty row in the summary grid.
- `slug(name: str) -> str` â€” Filename-safe form of a Session name ("Trial 1 Hands only" -> "Trial_1_Hands_only"), for the per-trial output HTML.
- `sniff_delimiter(path: Path) -> str` â€” Tab or comma, whichever actually splits this file into fields.
- `is_frame_number(cell: str) -> bool` â€” (no docstring)
- `load_trial_sessions(path: Path) -> dict[str, dict[str, str]]` â€” Read manual_frames.csv's (participant, session, CSV name) columns.
- `find_session_file(participant_dir: Path, csv_name: str) -> Path | None` â€” Resolve a manual_frames.csv ``CSV name`` cell to its file: the stem plus ".csv", nothing cleverer.
- `parse_frame_spec(spec: str) -> np.ndarray` â€” Parse "100-200,500,900-950" into a sorted unique 0-indexed int array.
- `load_trial_map(path: Path) -> dict[tuple[str, str], str]` â€” (no docstring)
- `find_trial_csv(participant: str, trial: str) -> tuple[Path | None, Path | None, Path | None]` â€” Return (trial csv, cascade-static csv, mano-calibration-static csv).
- `find_static_csv(participant: str) -> Path | None` â€” The participant's dedicated static/calibration recording, if any -- the *cascade-safe* one, not the MANO-calibration one (``find_trial_csv``'s third return value, used directly by MANO calibration callers).
- `find_manual_csv(vicon_csv: Path) -> Path | None` â€” The manually-labelled export next to ``vicon_csv``, if any.
- `find_manually_labelled_trials(data_root: Path=DATA_ROOT) -> list[tuple[str, str, Path, Path]]` â€” Every (participant, trial_stem, vicon_csv, manual_csv) pair on disk that has a manually-labelled export, across every participant directory.
- `auto_reference(markers: np.ndarray, need: int=300) -> np.ndarray` â€” Fallback reference frames: the longest fully-present stretches.

### `vicon2mano/core/emg.py`

OT Bioelettronica Novecento+ ``.otb4`` reader and per-channel HD-sEMG quality metrics.

- **class `Otb4Recording`** â€” Decoded ``.otb4``: raw int32 EMG per grid, ramps, and AUX channels (raw counts).
- `read_otb4(path: Path) -> Otb4Recording` â€” Stream an ``.otb4`` from its tar without extracting it to disk.
- `ramp_faults(ramp: np.ndarray) -> tuple[np.ndarray, np.ndarray]` â€” Sample indices where the ramp counter skipped (lost samples) or repeated.
- `windowed(x: np.ndarray, n: int) -> np.ndarray` â€” Reshape (C, T) to (C, T // n, n), dropping the ragged tail.
- `car_bandpass(raw: np.ndarray, fs: float, mv_per_count: float)` â€” (per-second p2p counts, dead mask, monopolar uV, CAR uV, CAR 20-450 Hz uV) of one (C, T) grid; dead rows NaN.
- `rms_envelope(raw: np.ndarray, fs: float, mv_per_count: float, win_s: float=0.05) -> np.ndarray` â€” (C, W) RMS in uV of the CAR 20-450 Hz signal over consecutive win_s windows.
- `channel_quality(raw: np.ndarray, fs: float, mv_per_count: float, win_s: float=0.5) -> dict[str, np.ndarray]` â€” Per-channel quality metrics for one (C, T) monopolar grid, plus the (C, W) windowed RMS in uV.
- `flag_channels(q: dict[str, np.ndarray]) -> list[str]` â€” Per-channel ';'-joined reasons a channel is unusable ('' = good).

### `vicon2mano/core/export.py`

Export FitResult to common formats.

- `save_npz(result: FitResult, path: str) -> None` â€” Save all parameters and joints to a compressed .npz file.
- `save_obj_sequence(result: FitResult, out_dir: str, *, every: int=1) -> None` â€” Write one .obj mesh file per frame (or every N frames) to out_dir.
- `to_dict(result: FitResult) -> dict` â€” Convert to a plain dict (e.g. for JSON serialisation).

### `vicon2mano/core/fitter.py`

Fit MANO parameters to Vicon 3D marker positions.

- `mano_output_to_joints21(out) -> 'torch.Tensor'` â€” Map an smplx MANO forward output to (B, 21, 3) in repo joint order.
- **class `FitConfig`** â€” (no docstring)
- **class `FitResult`** â€” Per-frame MANO parameters.
- **class `MANOFitter`** â€” Fit MANO to a sequence of Vicon 3D marker frames.
  - `.fit(markers: np.ndarray, marker_labels: list[str] | None=None, *, verbose: bool=True) -> FitResult` â€” Fit MANO to a sequence of marker frames.

### `vicon2mano/core/geometric_consistency.py`

Geometric self-consistency: does a trial's raw Vicon labelling produce anatomically plausible bone lengths, checked against nothing but its own geometry.

- `marker_bone_map(n_markers: int, bones: list[tuple[int, int]]) -> dict[int, list[int]]` â€” marker index -> list of bone indices (into ``bones``) touching it.
- `build_marker_frame_table(markers: np.ndarray, labels: list[str], bones: list[tuple[int, int, str]], ref_lengths: np.ndarray, *, subject_id: str, trial_id: st...)` â€” Long-format per-frame, per-marker geometric-deviation table.
- `add_threshold_columns(df: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame` â€” Add ``within_<t>mm`` boolean columns (deviation <= threshold).
- `flag_suspicious_events(df: pd.DataFrame, *, k: float=6.0, min_deviation_mm: float=4.0) -> pd.DataFrame` â€” Flag frames whose bone deviation is far above its marker's typical level: ``threshold = max(median(deviation) + k * MAD(deviation), min_deviation_mm)``.
- `geometric_events(df: pd.DataFrame) -> pd.DataFrame` â€” Collapse ``suspicious_flag`` rows into contiguous per-marker events.
- `missingness_events(df: pd.DataFrame, *, subject_id: str, trial_id: str) -> pd.DataFrame` â€” Contiguous per-marker runs of ``vicon_available == False`` (gaps).
- `per_marker_summary(df: pd.DataFrame, events: pd.DataFrame, gaps: pd.DataFrame, thresholds_mm: list[float]) -> pd.DataFrame` â€” One row per marker: availability, bone-deviation stats, event and gap counts -- the geometric-self-consistency analogue of ``vicon2mano.core.agreement.per_marker_summary`` / ``per_marker_event_summary`` /  ``missingness_events`` combined into one table, since the caller asked for a single per-mar...
- `trial_overall_summary(marker_summary: pd.DataFrame) -> dict` â€” Single-row trial-wide rollup of ``per_marker_summary`` (mean availability, worst median/p95 deviation, total events/gaps) -- what goes into the dataset-level CSV, one row per trial.
- `classify_reference_source(filename: str) -> tuple[str, bool]` â€” (reference_source_type, reference_source_is_static) from a filename.
- `reference_status(ref_lengths: np.ndarray) -> str` â€” Whether :func:`vicon2mano.core.bones.consensus_bone_lengths` produced a usable reference: ``"valid"`` (every bone got a finite length), ``"insufficient_bones"`` (some but not all -- e.g. a permanently-missing marker knocks out only the bones touching it, see ``vicon2mano.core.bones.consensus_bone...
- `bone_reference_table(markers: np.ndarray, bones: list[tuple[int, int, str]], ref_lengths: np.ndarray, inlier_frames: np.ndarray) -> pd.DataFrame` â€” Per-bone reference detail: how many of the consensus inlier frames actually had that specific bone finite, and the reference length that resulted -- the "number of valid frames used per bone" / "reference length per bone" figures, broken out per bone rather than only summarised trial-wide.

### `vicon2mano/core/loader.py`

Load Vicon marker data from .c3d or .csv files.

- `load_h5_trajectory(h5_path: str, participant: str, session: str) -> tuple[np.ndarray, list[str], float | None]` â€” Read one (participant, session) trajectory from the consolidated HDF5.
- `load_c3d(path: str) -> tuple[np.ndarray, list[str], float]` â€” Read a Vicon .c3d file.
- `load_csv(path: str, *, mm: bool=True) -> tuple[np.ndarray, list[str]]` â€” Read a Vicon-exported CSV (Nexus format).

### `vicon2mano/core/quality.py`

Raw Vicon quality assessment: classify every marker-frame using six separately-measured signals, without combining them into one score.

- `classify_marker_frame(is_missing: bool, manual_status: str | None, anchor_valid: bool, geometric_suspicious: bool) -> str` â€” One marker-frame's category, by precedence:
- `classify_vectorized(is_missing: np.ndarray, anchor_valid: np.ndarray, geometric_suspicious: np.ndarray, manual_category: np.ndarray) -> np.ndarray` â€” Vectorised form of :func:`classify_marker_frame` for a whole (T, N) array at once -- used by the production pipeline, where a Python-level loop over tens of millions of marker-frames is not practical. ``manual_category`` already has each reviewed cell's resolved category string (or "" where unrev...
- `resolve_manual_category(status: str | None) -> str` â€” Empty string (not None) for "unreviewed", matching :func:`classify_vectorized`'s sentinel convention for a pandas/numpy string array, which has no clean native null for object-dtype string comparisons at this scale.
- `summarize_counts(classification: pd.DataFrame) -> pd.DataFrame` â€” One row per (participant, trial_id, marker), with a count column per category in :data:`CATEGORIES` -- the aggregate ``raw_vicon_quality_summary.csv`` this module produces, since the full per-marker-frame table is tens of millions of rows dataset-wide and is kept per-trial instead (see ``scripts/...
- `adaptive_anchor_threshold(sine_values: np.ndarray, *, k: float=3.0, absolute_floor: float=0.1) -> float` â€” Lower bound on sine(Palm1-2-3 angle) for "valid", learned from one subject's own pooled distribution rather than assumed universal.

### `vicon2mano/core/validation.py`

Independent manual identity validation: compare hand-reviewed marker identities (from the 500-frame sample, labelled via ``label_selected_frames.m``) against the on-disk Vicon labels, and against the geometric self-consistency screen.

- `label_accuracy_summary(annotations: pd.DataFrame) -> dict` â€” Overall accuracy across the whole manual sample, plus the fraction of sampled *frames* (not individual marker judgements) that are fully confirmed correct.
- `per_marker_accuracy(annotations: pd.DataFrame) -> pd.DataFrame` â€” One row per marker (``marker_name``), same statistics as :func:`label_accuracy_summary` broken out per marker.
- `per_trial_accuracy(annotations: pd.DataFrame) -> pd.DataFrame` â€” One row per (participant, trial_id), same statistics as :func:`label_accuracy_summary` broken out per trial.
- `frame_geometric_status(events: pd.DataFrame, frame: int) -> tuple[bool, float]` â€” Whether ``frame`` falls inside any geometric-self-consistency event in ``events`` (that trial's ``events.csv``), and the largest ``deviation_magnitude_mm`` among events that cover it.
- `geometry_vs_manual_table(annotations: pd.DataFrame, events_by_trial: dict[tuple, pd.DataFrame]) -> pd.DataFrame` â€” Per-sampled-frame comparison table (one row per (participant, trial_id, frame)) with ``geometric_anomaly``, ``geometric_score``, ``trajectory_availability``, and ``manual_error_present`` -- the input to the 2x2 contingency table in the validation report.
- `geometry_vs_manual_contingency(per_frame: pd.DataFrame) -> pd.DataFrame` â€” The 2x2 (Normal/Anomalous x No/Yes) table from :func:`geometry_vs_manual_table`'s output.
- `classify_marker_frame(orig_valid: bool, corrected_valid: bool, same_coords: bool) -> str | None` â€” One (marker, frame) judgement inferred from a before/after coordinate diff, given a human deliberately reviewed that frame in Nexus.
- `to_palm_local_frame(markers: np.ndarray, labels: list[str], anchor_names: tuple[str, str, str]=DEFAULT_ANCHOR_NAMES) -> np.ndarray | None` â€” Re-express ``markers`` (T, N, 3) in a per-frame rigid frame anchored on 3 palm markers, cancelling out any *global* translation/rotation difference between two files -- found to be necessary, not optional, for comparing a raw Vicon CSV against a trajectory re-exported from a live Nexus session: m...
- `palm_triangle_bone_deltas(markers_orig: np.ndarray, labels_orig: list[str], markers_corrected: np.ndarray, labels_corrected: list[str], frame: int, anchor_na...)` â€” |orig bone length - corrected bone length| for each of the 3 pairs within the palm anchor triangle, at one frame -- the same check used in the audit's P7 spot-checks, generalised so it can be run systematically over every ``wrong_label`` judgement rather than by hand.
- `build_annotations_from_correction(markers_orig: np.ndarray, labels_orig: list[str], markers_corrected: np.ndarray, labels_corrected: list[str], visited_frame...)` â€” ``manual_annotations.csv`` rows for one trial, built by diffing the original Vicon-labelled recording against a hand-corrected export, for exactly the frames in ``visited_frames`` (everything else is unreviewed and must not be scored either way).

### `vicon2mano/core/video_sync.py`

Map RealSense frame timestamps onto a Vicon trial's clock.

- `vicon_start_ms(vicon_csv: Path) -> int` â€” Epoch ms (UTC) at which the Vicon capture started, from the trial's .xcp.
- `trigger_log_ms(rs_dir: Path) -> int | None` â€” Epoch ms of the first trigger_sent=true row of rs_dir's triggerlog, or None.
- `find_vicon_csv(rs_dir: Path) -> Path | None` â€” Vicon CSV for <root>/<P>/<Trial>/<cond>/RealSense, via <root>/manual_frames.csv.
- `read_sync_channel(vicon_csv: Path) -> tuple[np.ndarray, float] | None` â€” The Devices-section ``Sync`` analog channel and its rate (Hz); None for a trajectories-only export.
- `find_trigger_pulse(sig: np.ndarray, rate: float, width_s: float=1.0, tol: float=0.05) -> float | None` â€” Onset (s) of the first single high plateau lasting width_s +- tol, or None.
- `load_trajectories(vicon_csv: Path) -> tuple[np.ndarray, list[str], float]` â€” (markers (T, N, 3) mm, labels, Trajectories rate in Hz) of a Vicon trial CSV.
- `hand_speed(X: np.ndarray, labels: list[str], rate: float) -> tuple[np.ndarray, np.ndarray]` â€” (t [s], median hand-marker speed [mm/s]) between consecutive Vicon frames.
- `finger_speed(X: np.ndarray, labels: list[str], rate: float) -> np.ndarray` â€” Median finger-marker speed relative to the palm centroid [mm/s], running-median filtered over ~0.1 s.
- `motion_lag(video_t: np.ndarray, video_motion: np.ndarray, vicon_t: np.ndarray, vicon_speed: np.ndarray, search_s: float=3.0, step_s: float=0.005) -> tuple[fl...)` â€” (lag, r): the shift added to video_t that best correlates log-motion with log-speed.
- `sync_mask(sig: np.ndarray, fs: float, threshold: float, keep_s: float=0.2) -> np.ndarray` â€” 1 kHz boolean mask of where a sync channel is off its baseline (median) by more than threshold.
- `mask_offset(a: np.ndarray, v: np.ndarray) -> tuple[float, float, float]` â€” (lag s, IoU, ambiguity) aligning two 1 kHz sync masks so a(t + lag) ~ v(t).

### `vicon2mano/core/viz.py`

Shared visualisation vocabulary for relabelling animations.

- `marker_color(label: str) -> str` â€” (no docstring)
- `marker_digit(label: str) -> str` â€” (no docstring)
- `marker_bone_pairs(labels)` â€” (i, j) index pairs for MARKER_BONES present in this label set (subject prefixes like "P15:" ignored).

## Package: vicon2mano/strategies

### `vicon2mano/strategies/__init__.py`

Relabelling strategies. Each subpackage is one approach.


## Package: vicon2mano/strategies/cascade

### `vicon2mano/strategies/cascade/__init__.py`

Geometric marker-quality cascade.


### `vicon2mano/strategies/cascade/quality_cascade.py`

Marker-quality cascade: label every marker correct / incorrect / missing.

- `infer_bones(labels: list[str]) -> list[tuple[int, int, str]]` â€” Infer anatomically-adjacent marker pairs from label strings.
- `align_markers_to_labels(markers: np.ndarray, labels: list[str], target_labels: list[str]) -> np.ndarray` â€” Reorder/select marker columns from ``labels`` order to ``target_labels`` order, by exact (case-insensitive) name match.
- `build_reference(markers: np.ndarray, ref_frames: np.ndarray, bones: list[tuple[int, int, str]], *, extra_markers: np.ndarray | None=None) -> tuple[dict[tuple...)` â€” Reference bone lengths (median, MAD) and per-marker speed samples.
- `label_quality(markers: np.ndarray, labels: list[str], ref_frames: np.ndarray, *, static_markers: np.ndarray | None=None, bone_tol_mad: float=8.0, bone_tol_mm...)` â€” Return (T, N) status array: 0=missing, 1=incorrect, 2=correct.
- `label_quality_cascade(markers: np.ndarray, labels: list[str], ref_frames: np.ndarray, *, static_markers: np.ndarray | None=None, forearm_tol_mad: float=4.0, ...)` â€” Cascading trust-chain marker quality check.
- `debug_frame_cascade(markers: np.ndarray, labels: list[str], ref_frames: np.ndarray, t: int, *, static_markers: np.ndarray | None=None, forearm_tol_mad: float...)` â€” Print exactly how ``label_quality_cascade`` reasoned about frame ``t`` — every pairwise/anchor distance it checked, the reference it checked against, and whether each gate/marker passed. Same keyword arguments as ``label_quality_cascade`` (pass the same values you used to produce the status array...
- `main(argv: list[str] | None=None) -> None` â€” (no docstring)

## Package: vicon2mano/strategies/deep

### `vicon2mano/strategies/deep/__init__.py`

CNN marker labeler.


### `vicon2mano/strategies/deep/labeler.py`

CNN-based marker labeler.

- `voxelise_with_centroid(markers: np.ndarray, *, res: int=VOXEL_RES, half: float=SPATIAL_HALF, sigma: float=GAUSSIAN_SIGMA_VOXELS) -> tuple[torch.Tensor, np.nd...)` â€” Rasterise 3-D marker positions into a volumetric occupancy grid.
- **class `MarkerLabeler`** â€” 3-D U-Net for volumetric marker-to-joint heatmap regression.
  - `.forward(x: torch.Tensor) -> torch.Tensor` â€” 
- `assign_from_heatmaps(heatmaps: torch.Tensor, markers: np.ndarray, centroid: np.ndarray, *, res: int=VOXEL_RES, half: float=SPATIAL_HALF, conf_threshold: floa...)` â€” Decode 21 heatmap volumes into a marker index per joint.
- **class `DeepLabeler`** â€” Load trained weights and label Vicon marker frames.
  - `.label_frame(markers: np.ndarray) -> np.ndarray` â€” Label a single frame.
  - `.label_sequence(markers_seq: np.ndarray, *, batch_size: int=4) -> np.ndarray` â€” Label every frame in a sequence.

### `vicon2mano/strategies/deep/real_data.py`

Real labeled Vicon CSV loader for training the deep marker labeler.

- **class `RealDataConfig`** â€” (no docstring)
- **class `RealLabeledDataset`** â€” PyTorch-compatible dataset backed by a labeled Vicon CSV.
- `make_real_dataset(csv_path: str | Path, split: str='train', cfg: RealDataConfig | None=None) -> RealLabeledDataset` â€” Return a RealLabeledDataset for the given split.

### `vicon2mano/strategies/deep/synth_data.py`

Synthetic training data generator for the deep marker labeler.

- **class `SynthConfig`** â€” (no docstring)
- `build_target_heatmaps(joint_positions: np.ndarray, centroid: np.ndarray, *, res: int=64, half: float=0.22, sigma: float=1.5) -> np.ndarray` â€” Build (21, res, res, res) Gaussian heatmap targets.
- `augment_markers(joint_positions: np.ndarray, rng: np.random.Generator, cfg: SynthConfig) -> tuple[np.ndarray, int]` â€” Place, augment, and shuffle markers for one synthetic frame.
- `generate_dataset(cfg: SynthConfig | None=None) -> None` â€” Generate a synthetic dataset and write it to an HDF5 file.
- `make_synth_dataset(h5_path: str, split: str='train', val_frac: float=0.02)` â€” Return a SynthH5Dataset for the given split.

## Package: vicon2mano/strategies/gmm

### `vicon2mano/strategies/gmm/__init__.py`

strategyGMM: spatial GMM + Viterbi.


### `vicon2mano/strategies/gmm/labeler.py`

Spatial-GMM + Viterbi marker labeler ("strategyGMM").

- `rigid_frames(markers_mm: np.ndarray, origin_idx: int, x_idx: int, y_idx: int) -> tuple[np.ndarray, np.ndarray]` â€” Per-frame rigid orthonormal frame from 3 anchor marker indices.
- `to_local(markers_mm: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray` â€” World -> local. ``markers_mm``: (T, N, 3); ``R``/``origin``: (T, 3)/(T,3,3).
- `learn_anchor_triangle(markers_mm: np.ndarray, anchor_idxs: tuple[int, int, int], *, bin_mm: float=0.5, refine_mm: float=1.5) -> np.ndarray` â€” The 3 mutual distances of the anchor plate, learned from the data.
- `locate_anchor_triangle(markers_mm: np.ndarray, ref_d: np.ndarray, labelled: tuple[int, int, int], *, tol_mm: float=2.5) -> np.ndarray` â€” Find, per frame, which 3 markers actually form the anchor plate.
- `rigid_frames_from_triangle(markers_mm: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]` â€” Per-frame rigid frame built from a *per-frame* anchor triple.
- `mask_untrustworthy_frames(local_mm: np.ndarray, anchor_valid: np.ndarray) -> np.ndarray` â€” NaN out every marker's local position at frames with bad anchors.
- `to_world(local_mm: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray` â€” Local -> world, inverse of :func:`to_local`.
- **class `GMMParams`** â€” (no docstring)
- `fit_gmm_diag(samples: np.ndarray, *, n_components: int=3, n_iter: int=50, reg: float=1e-06, seed: int=0) -> GMMParams` â€” Fit a diagonal-covariance GMM via EM. Deterministic given ``seed``.
- `gmm_loglik(x: np.ndarray, gmm: GMMParams) -> np.ndarray` â€” Log-likelihood of point(s) ``x`` (..., 3) under ``gmm``.
- `fit_marker_gmms(local_mm: np.ndarray, marker_idxs: list[int], ref_frames: np.ndarray, *, n_components: int=3, n_iter: int=50, min_samples: int=20) -> dict[in...)` â€” One GMM per marker index, trained on ``ref_frames`` only.
- `loglik_matrix(obs_local: np.ndarray, gmms: dict[int, GMMParams], marker_order: list[int]) -> np.ndarray` â€” (M, K) log-likelihood matrix, ``M = len(marker_order)``.
- `top_n_assignments(cost: np.ndarray, n: int) -> list[tuple[np.ndarray, float]]` â€” Return up to ``n`` best assignments, best (lowest-cost) first.
- **class `MarkerKalman`** â€” Constant-velocity Kalman filter for one marker's local-frame position.
  - `.predict() -> np.ndarray` â€” Advance one step; returns the predicted position (3,).
  - `.residual_loglik(obs: np.ndarray) -> tuple[float, np.ndarray]` â€” Log-likelihood of ``obs`` under the current (already-predicted) state, without committing an update. Returns ``(loglik, residual)``.
  - `.gate_loglik() -> float` â€” Score assigned to *not* matching this filter to any observation this frame -- the log-density at the standard 95% chi-squared gate boundary (3 DOF, Mahalanobis distance^2 = 7.8147) of this filter's own current residual covariance, rather than a fixed constant.
  - `.update(obs: np.ndarray) -> None` â€” Innovation update. A non-finite observation is skipped, leaving the predicted state in place -- which is exactly the paper's occlusion handling (Section 3.2: extrapolate the trajectory from the predicted state, omit the innovation update), and keeps a single gap from permanently poisoning the fil...
  - `.copy() -> 'MarkerKalman'` â€” 
- `viterbi_select(marker_order: list[int], hypotheses_per_frame: list[list[tuple[np.ndarray, float]]], observations_per_frame: list[np.ndarray], init_positions:...)` â€” Pick the most probable sequence of hypotheses via Viterbi + Kalman.
- `relabel_local(local_mm: np.ndarray, gmms: dict[int, GMMParams], marker_idxs: list[int], *, n_hypotheses: int=5, theta_min: float=-30.0, process_var: float=1....)` â€” Hypothesis generation + Viterbi selection on precomputed local coords.
- `relabel_sequence(markers_mm: np.ndarray, labels: list[str], anchor_labels: tuple[str, str, str], ref_frames: np.ndarray, marker_idxs: list[int], *, n_hypothe...)` â€” Per-frame identity assignment for ``marker_idxs`` via GMM + Viterbi.
- **class `GMMLabeler`** â€” Fitted per-participant GMM model, reusable across trials.
  - `.fit(cls, markers_mm: np.ndarray, labels: list[str], anchor_labels: tuple[str, str, str], ref_frames: np.ndarray, marker_idxs: list[int], ...)` â€” See ``relabel_sequence`` for the ``anchor_valid`` contract.
  - `.relabel(markers_mm: np.ndarray, labels: list[str], *, n_hypotheses: int=5, theta_min: float=-30.0, process_var: float=1.0, obs_var: float...)` â€” 
- `segment_by_jumps(markers_mm: np.ndarray, idxs: tuple[int, int, int], *, jump_mm: float=3.0, min_len: int=30, bridge: int=5) -> list[np.ndarray]` â€” Split the recording into regimes of stable anchor geometry.
- `learn_reference_geometry(markers_mm: np.ndarray, idxs: tuple[int, int, int], segments: list[np.ndarray], *, tol_mm: float=2.0) -> np.ndarray` â€” The anchor geometry that recurs across the most *independent* regimes.
- `solve_segment_anchors(markers_mm: np.ndarray, segments: list[np.ndarray], ref_geom: np.ndarray, labelled: tuple[int, int, int], *, tol_mm: float=2.0, n_sampl...)` â€” Per-frame anchor triple, solved once per regime by pooling.

## Package: vicon2mano/strategies/mano

### `vicon2mano/strategies/mano/__init__.py`

MANO-model-based relabelling.


### `vicon2mano/strategies/mano/relabel.py`

Relabel *incorrectly labelled* markers using a subject-specific MANO model.

- `marker_joint_map(labels: list[str], side: str | None=None) -> dict[int, int]` â€” Map marker index -> MANO joint index, via the existing label seed.
- `verified_frames(status: np.ndarray, marker_idxs: list[int], *, max_frames: int=400) -> np.ndarray` â€” Frames where *every* marker in ``marker_idxs`` is CORRECT.
- `select_fit_frames(status: np.ndarray, marker_idxs: list[int], *, min_correct: int=12, max_frames: int=300, n_bins: int=30) -> np.ndarray` â€” Frames good enough to fit a pose on, spread across the recording.
- `fit_frames(markers_mm: np.ndarray, labels: list[str], frames: np.ndarray, *, mano_dir: str, side: str='right', betas: np.ndarray | None=None, n_iters_shape: ...)` â€” Fit MANO to an arbitrary set of frames. Returns the ``FitResult``.
- `calibrate_marker_offsets(markers_mm: np.ndarray, frames: np.ndarray, fit_joints_m: np.ndarray, m2j: dict[int, int], *, status: np.ndarray | None=None, marker...)` â€” Per-marker offset from its MANO joint, in that joint's *segment* frame.
- `predict_marker_positions(joints_m: np.ndarray, offsets: dict[int, np.ndarray], m2j: dict[int, int], n_markers: int) -> np.ndarray` â€” Expected position of every mapped marker, (F, n_markers, 3) metres.
- `model_reliability(markers_mm: np.ndarray, status: np.ndarray, predicted_m: np.ndarray, frames: np.ndarray, m2j: dict[int, int], *, stride: int=7, min_samples...)` â€” Can this trial's fitted model tell neighbouring markers apart?
- `relabel_frame(frame_markers_m: np.ndarray, predicted_m: np.ndarray, candidate_mask: np.ndarray, slot_mask: np.ndarray | None=None, *, max_dist_m: float=0.03)...)` â€” Reassign marker *positions* to the labels they best fit.
- `relabel_frame_iterative(markers_mm: np.ndarray, t: int, labels: list[str], m2j: dict[int, int], offsets: dict[int, np.ndarray], betas: np.ndarray, *, seed_ma...)` â€” Label-agnostic relabelling for one frame, reference/test implementation.
- `apply_relabel(markers: np.ndarray, frame: int, mapping: dict[int, int]) -> None` â€” Apply {slot: source} to one frame, in place. Positions are permuted between marker columns; no coordinate is ever modified.

## Package: vicon2mano/strategies/soma

### `vicon2mano/strategies/soma/__init__.py`

strategySOMA: learned, label-free per-frame marker labelling.


### `vicon2mano/strategies/soma/calibrate.py`

Calibrate the marker library to a new participant from their static.

- `is_forearm(name: str) -> bool` â€” (no docstring)
- `is_root(name: str) -> bool` â€” (no docstring)
- `static_cloud(markers: np.ndarray, n: int=200) -> tuple[np.ndarray, dict]` â€” (M, 3) median marker positions over the stillest, most complete ``n``-frame window of a static recording (NaN = never present there).
- `fit_static(model, lib: MarkerLibrary, cloud_mm: np.ndarray, *, device: str='cpu', iters_root: int=300, iters: int=1500, w_beta: float=2.0, w_pose: float=20.0...)` â€” Fit MANO shape and pose to one static cloud (mm). See module docstring.
- `calibrated_library(lib: MarkerLibrary, betas: np.ndarray, r_s: np.ndarray, miss_rate: np.ndarray | None=None, source: str='') -> MarkerLibrary` â€” Base library with the participant's shape and, for every marker with a finite static offset ``r_s``, residuals re-centred on it: ``residual[n, m] - median_n(residual[:, m]) + r_s[m]``.
- `final_static_library(lib: MarkerLibrary, model, params, cloud_mm: np.ndarray, miss_rate: np.ndarray | None=None, source: str='') -> MarkerLibrary` â€” ``lib`` re-attached at a final static fit's vertices (``scripts/soma/static_final_fit.py`` params.npz) with its shape and scale; each attached marker's residuals re-centred on its offset in ``cloud_mm`` (M, 3, lib order).

### `vicon2mano/strategies/soma/fill.py`

Learned marker completion: an MLP trained on synthetic clouds predicts every marker from the visible ones.

- `align(X: torch.Tensor, w: torch.Tensor, T: torch.Tensor)` â€” Weighted Kabsch of X (B, K, 3) onto template T (K, 3) with weights w (B, K): (R, cx, ct), canonical = (X - cx) @ R.mT + ct.
- **class `Filler`** â€” (no docstring)
  - `.forward(X: torch.Tensor, vis: torch.Tensor) -> torch.Tensor` â€” X (B, M, 3) mm (anything where not ``vis``), vis (B, M) bool -> every marker (B, M, 3) mm, same frame as X.
- `frame_sampler(X: np.ndarray, device: str='cpu')` â€” A ``train_filler`` source drawing random frames of real recordings X (N, M, 3) mm, NaN = unobserved.
- `train_filler(sample, labels: list[str], steps: int=20000, batch: int=1024, lr: float=0.001, log=print) -> Filler` â€” Train a Filler on clouds from ``sample(B) -> (X (B, M, 3) mm, present (B, M) bool)`` in ``labels`` order, hiding each marker with a per-cloud probability up to 0.5.
- `predict(f: Filler, X: np.ndarray, batch: int=8192) -> np.ndarray` â€” Every marker (N, M, 3) mm from the observed ones in X (N, M, 3) mm, NaN = unobserved; NaN rows where fewer than 3 rigid markers are seen.
- `save(f: Filler, path) -> None` â€” (no docstring)
- `load(path, device: str='cpu') -> Filler` â€” (no docstring)

### `vicon2mano/strategies/soma/infer.py`

Run a trained SOMA labeller over a whole raw trial.

- `load_model(path, device: str)` â€” (no docstring)
- `label_trial(net, raw: np.ndarray, n_points: int, device: str, batch: int=2048)` â€” Returns per_frame (T, C), conf (T, C), voted (T, C), piece ids (T, C).

### `vicon2mano/strategies/soma/library.py`

Marker library: where each marker sits on the MANO surface, plus real per-frame deviations from that spot.

- `marker_group(name: str) -> str` â€” Which pose source a marker follows: its finger, else the hand root.
- **class `MarkerLibrary`** â€” (no docstring)
  - `.subset(labels: list[str]) -> 'MarkerLibrary'` â€” The same library restricted to ``labels`` (in that order) -- e.g. a reduced marker set such as P1/P2's, where Forearm3, Forearm4 and Palm3 were never placed. Poses and every kept marker's geometry are unchanged.
  - `.save(path: str | Path) -> None` â€” 
  - `.load(cls, path: str | Path) -> 'MarkerLibrary'` â€” 
- `face_frames(V: np.ndarray, F: np.ndarray, face_idx: np.ndarray, bary: np.ndarray)` â€” Attachment points and local frames for every marker.
- `closest_point_on_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray` â€” Closest point on each triangle to p. p (3,), tri (K, 3, 3) -> (K, 3). Ericson, Real-Time Collision Detection 5.1.5, vectorised over K.
- `nearest_face(p: np.ndarray, V: np.ndarray, F: np.ndarray, vert_faces: list[np.ndarray], k_vert: int=12) -> int` â€” Index of the mesh face closest to point p (search near the k nearest vertices).
- `build_library(model, fit: dict, markers_mm: np.ndarray, real_mask: np.ndarray, labels: list[str], raw_present: np.ndarray, *, frames: np.ndarray, vote_stride...)` â€” See module docstring. ``max_fit_err_mm``: a marker whose fitted attachment point is further than this from the real marker in a frame gets a NaN residual there (bad fit -- don't copy it).

### `vicon2mano/strategies/soma/model.py`

SOMA-style labelling network: self-attention over the cloud + Sinkhorn.

- **class `SomaLabeler`** â€” (no docstring)
  - `.forward(pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor` â€” pts (B, N, 3) mm, valid (B, N) bool -> log assignment (B, N+1, M+1).
- `log_sinkhorn(scores: torch.Tensor, alpha: torch.Tensor, valid: torch.Tensor, iters: int) -> torch.Tensor` â€” Log-space Sinkhorn with a dustbin row and column, padding-aware.
- `assignment_nll(logP: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor` â€” Mean NLL of the true assignment. label (B, N): marker index, -1 ghost, -2 pad.
- `hard_assign(logP: np.ndarray, n_valid: int) -> np.ndarray` â€” One-to-one labels from a (N+1, M+1) log assignment for one cloud.

### `vicon2mano/strategies/soma/realdata.py`

Real Vicon frames as unlabelled clouds, with ground-truth identities.

- `raw_identities(raw: np.ndarray, trusted: np.ndarray, tol_mm: float=1.0) -> np.ndarray` â€” (T, C) identity of every raw column per frame: label index into the trusted columns, -1 ghost / unmatched, -2 absent.
- `clouds(raw: np.ndarray, ident: np.ndarray, frames: np.ndarray, n_points: int)` â€” Padded, median-centred clouds for ``frames``.
- `pieces(raw: np.ndarray, jump_mm: float=5.0) -> np.ndarray` â€” (T, C) piece id per raw slot (-1 absent): each column is cut wherever the marker disappears or moves more than ``jump_mm`` in one frame.
- `vote_pieces(frame_label: np.ndarray, frame_conf: np.ndarray, pid: np.ndarray, n_labels: int) -> np.ndarray` â€” Give every piece the label with the largest summed confidence over its frames, then resolve clashes: where two slots in a frame share a label, the piece with the larger vote share keeps it and the other is unlabelled (-1) in that frame. frame_label/conf: (T, C) per-frame labels (-1 none).
- `relabelled_markers(raw: np.ndarray, slot_label: np.ndarray, n_labels: int) -> np.ndarray` â€” (T, n_labels, 3): each raw point moved to the column of its assigned label. Unlabelled points (-1) are dropped; positions are never altered.
- `write_wide_csv(path, markers: np.ndarray, labels: list[str]) -> None` â€” (T, N, 3) mm -> single-header wide Nexus CSV that ``vicon2mano.core.loader.load_csv`` reads back; blank cells for NaN.

### `vicon2mano/strategies/soma/synth.py`

Synthetic marker clouds for training the labeller, generated on the fly.

- **class `SynthConfig`** â€” (no docstring)
- **class `SynthGenerator`** â€” (no docstring)
  - `.n_points() -> int` â€” 
  - `.sample(B: int, *, mix: bool=True, augment: bool=True, src: torch.Tensor | None=None, complete: bool=False)` â€” Generate B clouds. ``mix=False, augment=False`` with ``src`` given reproduces library frames exactly (round-trip test). ``complete``: return the labelled markers instead, (B, M, 3) mm in the hand's own frame plus (B, M) which of them carry a measured residual -- before drops, ghosts, rotation and...
- `random_rotations(B: int, g: torch.Generator, full: bool=True) -> torch.Tensor` â€” (B, 3, 3) uniform rotations (Shoemake quaternions), or about z only.

## Package: vicon2mano/strategies/template

### `vicon2mano/strategies/template/__init__.py`

Template/distance-descriptor matching.


## Top level

### `data/h5_trajectory_reader.py`

h5_trajectory_reader.py ====================== Utility for reading and querying the consolidated HDF5 trajectory file.

- **class `H5TrajectoryReader`** â€” Reader for consolidated trajectory HDF5 file.
  - `.close() -> None` â€” Close the HDF5 file.
  - `.list_participants() -> list[str]` â€” Get sorted list of all participants.
  - `.list_sessions(participant: str) -> list[str]` â€” Get sorted list of sessions for a participant.
  - `.list_all() -> dict[str, list[str]]` â€” Get all participants and their sessions.
  - `.get_trajectory(participant: str, session: str, as_dict: bool=False) -> pd.DataFrame | dict` â€” Read a specific trajectory.
  - `.get_participant_trajectories(participant: str) -> dict[str, pd.DataFrame]` â€” Read all trajectories for a participant.
  - `.get_session_type(session_pattern: str) -> dict[str, dict[str, pd.DataFrame]]` â€” Read all trajectories matching a session type (substring match).
  - `.get_stats() -> dict` â€” Get file statistics.
  - `.print_summary() -> None` â€” Print a summary of the HDF5 file.
- `main() -> None` â€” (no docstring)

### `visualize_data.py`

Debug-friendly script version of visualize_data.ipynb.

- `load_trial_filename_map(path: Path) -> dict[tuple[str, str], str]` â€” Read trial_filename_map.csv: participant,filename,trial_key.
- `find_trial_files(participant: str, participant_dir: Path, overrides: dict[tuple[str, str], str] | None=None) -> dict[str, Path]` â€” Map canonical trial keys ("Trial1_handsonly", ..., and "static" for a calibration/static recording if present) to actual CSV files in a participant's directory.
- `auto_reference(markers, min_frames=300)` â€” Fallback when a trial has no row in manual_frames.csv yet: the longest run(s) of frames where every marker is present. Unverified -- prefer adding a real reference range to manual_frames.csv instead.
- `trial_stats(participant, session, csv_file, markers, status, n_ref)` â€” One row of before/after numbers for a single trial.
- `marker_color(label)` â€” (no docstring)
- `marker_digit(label)` â€” Trailing digit(s) of a marker label ("Thumb1" -> "1"), for a compact on-plot annotation next to each marker dot.
- `marker_trace_data(markers, status, t)` â€” (no docstring)
- `bone_lines(frame_markers, bones)` â€” (x, y, z) with None gaps so all bone segments render as one trace.
- `build_animation_figure(participant, trial, markers, labels, bones, status, pct_correct)` â€” Build the interactive 3D Plotly animation figure for one trial.
- `save_animation_html(fig, n_frames, out_path)` â€” (no docstring)

## Scripts: scripts/cascade

### `scripts/cascade/run_quality_cascade.py`

CLI for the marker-quality cascade.


## Scripts: scripts/deep

### `scripts/deep/eval_labeler.py`

Post-training evaluation: training curves, per-joint accuracy, and 3-D Vicon/MANO overlays.

- `load_checkpoint(path: str, device: torch.device) -> tuple[MarkerLabeler, dict]` â€” (no docstring)
- `load_log(weights_path: str) -> list[dict] | None` â€” (no docstring)
- `evaluate_frames(model: MarkerLabeler, frames: list[tuple[np.ndarray, np.ndarray]], device: torch.device, max_frames: int | None=None) -> list[dict]` â€” Run the labeler on every val frame (or up to max_frames).
- `plot_training_curves(log: list[dict], meta: dict, out_dir: Path | None)` â€” (no docstring)
- `plot_joint_performance(results: list[dict], out_dir: Path | None)` â€” (no docstring)
- `plot_3d_overlays(results: list[dict], n_frames: int, out_dir: Path | None)` â€” (no docstring)
- `plot_mano_overlay(results: list[dict], mano_dir: str, n_frames: int, out_dir: Path | None)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/deep/train_labeler.py`

Train the deep marker labeler (Han et al., SIGGRAPH 2018).

- `focal_bce_loss(logits, targets, *, gamma: float=2.0, alpha: float=0.75)` â€” Sigmoid focal binary cross-entropy.
- `main()` â€” (no docstring)

## Scripts: scripts/emg

### `scripts/emg/emg_pane.py`

Add a trial's HD-sEMG to its RealSense player: per grid, a 16x6 electrode map and a channel x time heatmap.

- `emg_sync(aux: np.ndarray, fs: float, sync: dict) -> dict | None` â€” EMG-minus-Vicon offset: AUX 1 vs Vicon Sync pattern, else AUX 1's 1 s pulse vs the RealSense trigger log.
- `main()` â€” (no docstring)

### `scripts/emg/emg_quality.py`

Per-channel quality report for the Novecento+ HD-sEMG recordings.

- `aux_bursts(aux: np.ndarray, fs: float, gap_s: float=0.5) -> list[tuple[float, float]]` â€” (start, end) seconds of each interval where AUX1 (the Vicon sync input) is off baseline.
- `run(otb4: Path) -> list[dict]` â€” Analyse one recording; write its channel CSV and heat map, return its summary rows.
- `main() -> None` â€” (no docstring)

## Scripts: scripts/gmm

### `scripts/gmm/_common.py`

Shared loading/fitting logic for validate_gmm_labeler.py and animate_gmm_relabel.py, factored out once both needed it to avoid the two scripts silently drifting apart on how they compute anchor validity, reference frames, etc.

- **class `TrialContext`** â€” (no docstring)
- `load_p7_ring1_pinky1(*, buf: int=500, max_ref: int=3000) -> TrialContext` â€” (no docstring)
- `fit_and_relabel_window(ctx: TrialContext, lo: int, hi: int, *, n_hypotheses: int=5, process_var: float=DEFAULT_PROCESS_VAR, obs_var: float=DEFAULT_OBS_VAR) -...)` â€” Fit on ``ctx.ref_frames`` (anchor-gated), relabel window [lo, hi).

### `scripts/gmm/animate.py`

Animate the before/after effect of strategyGMM's relabelling.

- `render_before_after(ctx, before: np.ndarray, after: np.ndarray, lo: int, is_swapped: np.ndarray, out_stem, title: str, *, truth: np.ndarray | None=None, fps:...)` â€” Side-by-side before/after 3-D animation -> ``.gif`` and ``.html``.
- `main() -> None` â€” (no docstring)

### `scripts/gmm/relabel_trial.py`

Whole-trial strategyGMM relabelling + animation, with no cascade.

- `load_prior_gmms(prior_csv: Path, target_by_name: dict[str, int]) -> dict[int, gl.GMMParams]` â€” Marker GMMs from a separately labelled trial assumed fully correct.
- `run(participant: str, trial: str, *, n_out: int, n_hypotheses: int, tol_mm: float, passes: int, process_var: float, obs_var: float, out_dir: Path, min_persis...)` â€” (no docstring)
- `build_figure(corrected, original, labels, bones, status, found, as_labelled, frame_idx, participant, trial, err_before, err_after)` â€” Plotly animation matching results/mano style.
- `main(argv=None)` â€” (no docstring)

### `scripts/gmm/validate.py`

Real-data validation of strategyGMM against a known, documented swap.

- `main() -> None` â€” (no docstring)

### `scripts/gmm/validate_injected_swap.py`

Ground-truth validation of strategyGMM: inject a known swap into clean real marker data and measure whether it is recovered.

- `main() -> None` â€” (no docstring)

## Scripts: scripts/mano

### `scripts/mano/animate_mesh.py`

HTML animation: fitted MANO mesh surface + Vicon marker overlay, at real elapsed-time playback speed.

- `resolve_model_path(mano_dir: str, side: str) -> Path` â€” (no docstring)
- `forward_vertices(model, global_orient, hand_pose, transl, betas, scale=1.0, thickness=1.0, seglen=None) -> np.ndarray` â€” No-grad forward pass -> (n, 778, 3) mesh vertices in metres, with the fit's global ``scale``, palm ``thickness`` and per-segment finger lengths ``seglen`` (fit_surface_markers --fit-scale / --fit-thickness / --fit-segments).
- `load_model(mano_dir: str, side: str, pose_dim: int)` â€” smplx MANO matching a fit's pose parameterisation: 45 = raw axis-angle, else PCA.
- `fit_vertices_mm(model, fit, frame_idx) -> np.ndarray` â€” (n, 778, 3) mesh vertices in mm for these frames of a FitResult npz.
- `view_box(verts_mm, markers_mm)` â€” (center, half-width) of a fixed cubic view over every vertex and finite marker.
- `render_mesh_frames(verts_mm, markers_mm, faces, bone_pairs, center, half, frame_numbers, dpi=85) -> list[bytes]` â€” One JPEG per frame: opaque mesh + marker bones + markers; center is (3,) for fixed axes or (n, 3) to follow.
- `main() -> None` â€” (no docstring)

### `scripts/mano/calibrate_shape_on_static.py`

Calibrate MANO shape (betas) on a static trial, then fit pose with it frozen.

- `med_dist(arr, base, a, b)` â€” (no docstring)
- `screen_static(st, bs, mv, bm, tol=5.0)` â€” Print static-vs-movement agreement per segment; return the bad names.
- `main() -> None` â€” (no docstring)

### `scripts/mano/export_joint_angles.py`

Export per-finger joint angles from a completed MANOFitter result.

- `decode_full_pose(hand_pose: np.ndarray, model) -> np.ndarray` â€” (T, n_pca) or (T, 45) hand_pose -> (T, 15, 3) axis-angle per joint.
- `joint_angles_deg(hand_pose: np.ndarray, model) -> pd.DataFrame` â€” (no docstring)
- `plot_finger(df: pd.DataFrame, finger: str, time_s: np.ndarray, out_path: Path, title_suffix: str) -> None` â€” (no docstring)
- `main() -> None` â€” (no docstring)

### `scripts/mano/fit_single_frame.py`

Fit MANO to a short window of frames and plot one highlighted frame.

- `main() -> None` â€” (no docstring)

### `scripts/mano/fit_surface_markers.py`

Fit MANO so the markers sit ON THE MESH SURFACE, not at joint centres.

- `vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray` â€” Area-weighted vertex normals for a triangle mesh.
- `allowed_faces(lbs, F, repo_joint, min_own=0.0)` â€” Faces whose skin belongs to the bone a marker is taped to.
- `joint_limit_bounds()` â€” (45,) lower and upper bounds on the raw axis-angle pose.
- `joint_limit_penalty(hp, lo, hi)` â€” Squared hinge loss for any pose component outside its range.
- `joint_axis_basis(model, dev, pose_dim)` â€” Per-joint orthonormal basis (flexion, abduction, twist) in POSE space.
- `basis_limit_bounds(torsion_only=False, free_flexion=False)` â€” (15,3) lo/hi in the measured per-joint basis.
- `basis_limit_penalty(hp, Bt, lo, hi, twist_w=1.0)` â€” Hinge penalty on the projection onto each joint's measured basis.
- `run_adam(opt, loss_fn, n_iters, graph=False, label='', post_grad=None, report=None, check_every=100)` â€” Run ``n_iters`` Adam steps of ``loss_fn()``; return ``report()``.
- `angles_from_markers(mk, names, Bmat, gap_fill=True)` â€” Initial hand_pose read straight off the markers, per frame.
- `stretch_offsets(J21, hp, Bt, names, slopes, ref_deg=None, gate=45.0)` â€” Distal slide of each attachment, as a function of fitted flexion.
- `segment_frames_t(J21)` â€” (T, P, 3, 3) per-joint frames aligned to each joint's own bone, in torch.
- `calibrate_segment_offsets(J21, M, names, repo_of)` â€” Per-marker offset in its joint's segment frame -- the ALTERNATIVE to a mesh attachment, ported from scripts/mano/refit_with_marker_offsets.py.
- `thumb_slide(J21, names, slide)` â€” (T, P, 3) metres: each thumb marker's constant slide along its segment.
- `onehot_attachment(F, vid)` â€” Express "attach to vertex ``vid``" as a (face, barycentric) pair.
- `palm_row_loss(spec, mk, names, F, dev)` â€” Loss on where the mesh's skin edges lie along rows across the palm (from a photo, photo_overlay.py).
- `web_loss(spec, dev)` â€” Loss on where the mesh's finger-web vertices land in the photo (photo_overlay.py's camera and detected webs).
- `write_vertex_picker(V, F, mk, names, path)` â€” Mesh + every vertex id, hoverable, so attachments can be read off.
- **class `SegmentModel`** â€” MANO with a length factor per finger segment (15 bones); log-length 0 = MANO's.
- **class `ScaledModel`** â€” MANO with one global scale about the wrist (smplx has none); log-scale 0 = unscaled.
- `dorsal_direction(J, palm_marker, offset=None)` â€” Unit vector pointing out the BACK of the hand (palm-level).
- `segment_dorsals(J, palm_dorsal, repo_joints, thumb_mode='cross')` â€” Per-marker dorsal direction that rotates with the finger.
- `closest_point_on_faces(V, F, pts, allow=None, dorsal=None, min_dot=0.0)` â€” Nearest point on the mesh *surface* for each of ``pts``.
- `surface_report(V, F, mk, names, tag)` â€” Signed distance marker -> nearest vertex (+ outside / - inside).
- `main() -> None` â€” (no docstring)

### `scripts/mano/refit_with_marker_offsets.py`

Two-pass MANO fit: calibrate per-marker skin offsets, then refit pose against offset-corrected targets instead of raw marker positions.

- `main() -> None` â€” (no docstring)

### `scripts/mano/relabel_trial.py`

Relabel incorrectly-labelled markers with a subject-specific MANO model.

- `marker_color(label: str) -> str` â€” (no docstring)
- `marker_digit(label: str) -> str` â€” (no docstring)
- `relabel_trial(participant: str, trial: str, *, min_correct_pct: float, n_out: int, min_correct_calib: int, max_dist_mm: float, min_trusted: int=6, side: str=...)` â€” Fit, predict, then assign at one gate value. See _prepare_trial.
- `build_figure(d, participant, trial, min_correct_pct)` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)
- `run_one(participant, trial, args)` â€” (no docstring)
- `run_all(args)` â€” Relabel every (participant, trial) in manual_frames.csv.

### `scripts/mano/relabel_trial_loop.py`

Label-agnostic MANO relabelling — thin preset over ``relabel_with_mano.py``.

- `main(argv=None)` â€” (no docstring)

### `scripts/mano/score_fit_vs_reference.py`

Score a fit against a reference fit (e.g. the manual-label fit), split by whether Vicon's labels were right or wrong *for that finger* in each frame.

- `finger_of(label: str) -> str | None` â€” (no docstring)
- `finger_frame_class(raw, manual, raw_names, man_names)` â€” {finger: 'right'|'wrong'|'other' per frame} from Vicon-vs-manual labels.
- `load_pose(path, model_dir=None)` â€” (no docstring)
- `geodesic_deg(a, b)` â€” (no docstring)
- `fingers_residual(log: Path | None)` â€” (no docstring)
- `main()` â€” (no docstring)

## Scripts: scripts/shared

### `scripts/shared/aggregate_manual_agreement.py`

Run the single-trial agreement analysis across every trial that has a manually-labelled export, then roll the per-marker reliability numbers up across trials/participants.

- `write_report(out_dir: Path, all_reliability: pd.DataFrame, cross_summary: pd.DataFrame, trials: list[tuple[str, str, Path, Path]]) -> None` â€” (no docstring)
- `run(out_dir: Path, thresholds_mm: list[float], jump_mad_multiplier: float, jump_floor_mm: float, per_trial_plots: bool, per_trial_reports: bool) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/analyze_manual_agreement.py`

Quantify agreement between on-disk Vicon labels and a manually-labelled export, for one complete trial.

- `make_plots(df, marker_summary, out_dir: Path) -> None` â€” (no docstring)
- `write_report(out_dir: Path, participant, trial, vicon_csv, manual_csv, match, overall, marker_summary, events, event_summary, missingness, reliability) -> None` â€” (no docstring)
- `run(participant: str | None, trial: str | None, vicon_csv: Path | None, manual_csv: Path | None, out_dir: Path, thresholds_mm: list[float], jump_mad_multipli...)` â€” Run the full single-trial analysis, writing every output to ``out_dir``. Returns the per-marker reliability table (see ``agreement.reliability_table``) so callers that need to aggregate across trials (``aggregate_manual_agreement.py``) don't have to re-parse the CSVs or re-derive it themselves.
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/animate_fit.py`

Side-by-side animation: raw Vicon recording vs fitted/corrected MANO hand.

- `main()` â€” (no docstring)

### `scripts/shared/build_annotations_from_correction.py`

Build ``manual_annotations.csv`` (deliverable #2) by diffing each trial's original Vicon-labelled CSV against a hand-corrected export you made by relabelling directly in Nexus -- the alternative to typing per-marker verdicts into ``label_selected_frames.m``.

- `find_visited_csv(explicit: Path | None) -> Path` â€” (no docstring)
- `run(visited_csv: Path, reviewer_override: str | None, out_csv: Path, atol_mm: float) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/build_raw_vicon_quality_summary.py`

Phase 1B: label and frame-quality assessment across the whole dataset.

- `trial_dir(participant: str, trial_id: str) -> str` â€” (no docstring)
- `anchor_sine(markers: np.ndarray, labels: list[str]) -> np.ndarray` â€” (T,) float: sine of the Palm1-2-3 angle at Palm1 each frame -- NaN where the triple is absent from this file or not all finite that frame. Split out from the old combined anchor_validity() so a threshold (fixed or per-subject adaptive, see core.quality.adaptive_anchor_threshold) can be applied af...
- `anchor_valid_from_sine(sine: np.ndarray, threshold: float) -> np.ndarray` â€” (T,) bool from anchor_sine's output and a (fixed or per-subject adaptive) lower bound. NaN (anchor absent/missing that frame) is always invalid, regardless of threshold.
- `expand_events_to_bool(events: pd.DataFrame, marker_base_names: list[str], T: int) -> dict` â€” {base_marker_name: (T,) bool} -- True where that marker falls inside any geometric-self-consistency event (full [start_frame, end_frame] span), via interval accumulation rather than a per-frame Python loop.
- `event_detail_lookup(events: pd.DataFrame) -> dict` â€” {(base_marker_name, frame): (event_type, associated_bone)} for every frame actually covered by an event -- used to carry the "suspicious swap event" detail columns without re-deriving them.
- `find_all_trials(sessions: dict, participants: list[str]) -> list[tuple[str, str, Path]]` â€” (no docstring)
- `run(out_dir: Path, k: float, absolute_floor: float, write_per_trial_detail: bool) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/build_sampling_manifest.py`

Build ``sampling_manifest.csv`` (deliverable #1 of the manual-validation plan): turn the 500-frame anchor selection into a MATLAB-ready review list, resolving each frame's actual trial CSV, and where a frame was selected *because of* a specific marker/bone (largest_anomaly, geometry_availability_...

- `trial_dir_name(participant: str, trial_id: str) -> str` â€” (no docstring)
- `load_trial_metadata(participant: str, trial_id: str) -> dict | None` â€” (no docstring)
- `load_events_lookup(participant: str, trial_id: str) -> pd.DataFrame | None` â€” (no docstring)
- `build_review_list(anchors: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/build_trial_overview.py`

One row per trial, pulling together geometric self-consistency metrics that are otherwise spread across ``results/shared/geometric_consistency/``.

- `classify_trial_type(trial_id: str) -> str` â€” "Hands only" vs "HOI" (hand-object interaction), read off the session name exactly as manual_frames.csv spells it -- same two categories CLAUDE.md's own state-of-the-data notes use ("hands only trials are consistently 2-10x cleaner than HOI").
- `build_overview(trial_summary: pd.DataFrame) -> pd.DataFrame` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/build_validation_audit.py`

One-shot validation audit: accuracy/coverage headline numbers, breakdowns by trial/marker/sampling-stratum, why frames came out "ambiguous", direct P7 spot-checks, and anchor-frame validity diagnostics for the manual_identity_validation assessment (results/shared/frame_sample/ manual_annotations....

- `anchor_degeneracy(markers: np.ndarray, labels: list[str]) -> np.ndarray | None` â€” Per-frame sine-like degeneracy measure of the Palm1-2-3 triangle: ``|cross(x, y_raw)| / (|x| * |y_raw|)``, 0 for exactly collinear anchors, ~1 for a well-formed right angle. ``rigid_frames`` normalises its cross product with a ``+1e-12`` guard, so a collinear triangle produces a *finite*, numeric...
- `trial_dir(participant: str, trial_id: str) -> str` â€” (no docstring)
- `load_pair(participant: str, trial_id: str)` â€” (no docstring)
- `run(annotations_csv: Path, manifest_csv: Path, out_dir: Path, degenerate_threshold: float, spotcheck_participant: str, n_spotchecks: int) -> None` â€” (no docstring)
- `df_to_md(df: pd.DataFrame, float_cols: tuple=(), floatfmt: str='{:.3f}') -> str` â€” Minimal GitHub-flavoured markdown table -- avoids a hard dependency on the ``tabulate`` package that ``DataFrame.to_markdown`` needs, which isn't in this repo's dependencies (same approach analyze_manual_agreement.py already uses).
- `write_report(out_dir, overall, determinate_coverage, trial_acc, marker_acc, stratum_acc, unmatched, cause_counts, anchor_diag, degenerate_examples, spotcheck...)` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/build_validation_report.py`

Build the final 4 outputs of the manual-validation plan from the 500-frame sample's completed ``manual_annotations.csv``:

- `find_annotations_csv(explicit: Path | None) -> Path` â€” (no docstring)
- `trial_dir_name(participant: str, trial_id: str) -> str` â€” (no docstring)
- `load_events_by_trial(pairs) -> dict[tuple, pd.DataFrame]` â€” (no docstring)
- `load_manifest() -> pd.DataFrame | None` â€” (no docstring)
- `validate_annotation_columns(annotations: pd.DataFrame) -> None` â€” (no docstring)
- `build_html_report(out_dir: Path, annotations: pd.DataFrame, manifest: pd.DataFrame | None, overall: dict, marker_summary: pd.DataFrame, trial_summary: pd.Dat...)` â€” (no docstring)
- `build_p10_section(annotations: pd.DataFrame) -> str` â€” (no docstring)
- `run(annotations_csv: Path, out_dir: Path) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/compare_relabel_csvs.py`

Compare a raw Vicon export against manually-relabelled version(s) of it.

- `short_label(label: str) -> str` â€” (no docstring)
- `load_all(raw_path: Path, compare_paths: list[Path])` â€” (no docstring)
- `classify_changes(raw: np.ndarray, cmp: np.ndarray) -> dict` â€” Per-marker stats for `cmp` vs `raw`. raw, cmp: (T, N, 3).
- `bone_lengths(markers: np.ndarray, bones: list[tuple[int, int, str]]) -> np.ndarray` â€” (T, n_bones) length of each bone per frame; NaN where either end is missing.
- `print_bone_report(name: str, bone_names: list[str], raw_len: np.ndarray, cmp_len: np.ndarray) -> None` â€” (no docstring)
- `make_bone_plots(name: str, bone_names: list[str], raw_len: np.ndarray, cmp_len: np.ndarray, out_dir: Path) -> None` â€” (no docstring)
- `print_report(name: str, labels: list[str], stats: dict) -> None` â€” (no docstring)
- `make_plots(name: str, labels: list[str], stats: dict, out_dir: Path) -> None` â€” (no docstring)
- `make_missing_plot(datasets, out_dir: Path) -> None` â€” (no docstring)
- `main() -> None` â€” (no docstring)

### `scripts/shared/detect_swaps.py`

Detect (and optionally fix) marker-label swaps in "clean" recordings.

- `side_markers(markers, labels, side)` â€” (no docstring)
- `list_recordings(h5, session_type)` â€” (no docstring)
- `build_consensus(h5, recordings, side, stride, ref_labels=None)` â€” (no docstring)
- `plausible_pairs(canon, labels, k=4)` â€” k nearest labels by descriptor, restricted to finger markers.
- `detect_swaps(markers, labels, canon, bone_canon, pairs, *, stride, margin)` â€” Return [(i, j, gain, confidence)] swaps; confidence high if bone OR motion confirms the descriptor flag, else medium.
- `validate(h5, recordings, side, stride, margin, canon, bone_canon, labels_ref)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/shared/eval_fit.py`

Visualise MANOFitter results: residuals and 3-D marker/skeleton overlays.

- `per_joint_residuals(joints: np.ndarray, markers_m: np.ndarray, assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]` â€” Return (T, 21) per-frame residuals in mm (NaN where unassigned/missing) and the (21,) marker index per joint.
- `plot_residuals(res: np.ndarray, assignment: np.ndarray, labels, out_dir)` â€” (no docstring)
- `plot_residual_time(res: np.ndarray, out_dir)` â€” (no docstring)
- `plot_overlays(joints, markers_m, assignment, n_frames, out_dir)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/shared/eval_synth_fit.py`

Synthetic ground-truth evaluation of the MANO fitting pipeline.

- `generate_sequence(rng, model, T, offset_m, noise_m, device, gt_pca=GT_PCA, protocol=None)` â€” Return (markers_mm (T, 16, 3), labels, gt_joints (T, 21, 3) metres).
- `main()` â€” (no docstring)

### `scripts/shared/export_finger_angles.py`

Export per-frame finger flexion angles to CSV.

- `flexion_angles(joints: np.ndarray) -> dict[str, np.ndarray]` â€” joints: (T, 21, 3) metres -> {f"{finger}_{joint}": (T,) degrees}.
- `main()` â€” (no docstring)

### `scripts/shared/gen_api_reference.py`

- `first(doc, n=2)` â€” (no docstring)
- `sig(fn)` â€” (no docstring)
- `section(f)` â€” (no docstring)

### `scripts/shared/geometric_self_consistency.py`

Geometric self-consistency report: does a trial's own bone-length geometry hang together, checked against nothing but itself.

- `analyze_trial(markers: np.ndarray, labels: list[str], *, subject_id: str, trial_id: str, analysis_csv_name: str, thresholds_mm: list[float], jump_mad_multipl...)` â€” (no docstring)
- `make_plots(df: pd.DataFrame, marker_summary: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> None` â€” (no docstring)
- `write_trial_report(out_dir: Path, overall: dict, marker_summary: pd.DataFrame, events: pd.DataFrame, gaps: pd.DataFrame, reference_table: pd.DataFrame) -> None` â€” (no docstring)
- `run_one(participant: str, trial: str, out_dir: Path, thresholds_mm: list[float], jump_mad_multiplier: float, min_deviation_mm: float, make_plots_flag: bool)` â€” (no docstring)
- `write_global_report(out_dir: Path, all_marker_summary: pd.DataFrame, trial_summaries: pd.DataFrame, failures: list[tuple[str, str, str]]) -> None` â€” (no docstring)
- `run_all(out_dir: Path, thresholds_mm: list[float], jump_mad_multiplier: float, min_deviation_mm: float, make_plots_flag: bool) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/realsense_player.py`

Export RealSense .bag colour streams to MP4, sync them to Vicon, build an HTML player.

- `read_color_frames(bag: Path)` â€” Yield (frame_number, global_timestamp_ms, rgb array) once per recorded colour frame.
- `export_bag(bag: Path, mp4: Path, crf: int=23) -> list[tuple]` â€” Write one bag's colour stream to a VFR MP4; return (frame_number, ts_ms, pts_ms, motion) rows.
- `cam_summary(name: str, rows) -> dict` â€” Frame-rate stats plus the arrays the player needs.
- `sync_to_vicon(vicon_csv: Path, rs_dir: Path, cams: list[dict], offset: float | None, X: np.ndarray, labels: list[str], rate: float) -> tuple[dict, np.ndarray]` â€” Measure the RealSense->Vicon offset; return (sync record, hand-speed trace at 50 Hz).
- `to_50hz(x: np.ndarray, rate: float) -> np.ndarray` â€” Block-mean a Vicon-rate trace down to 50 Hz for the player.
- `markers_pane(X: np.ndarray, labels: list[str], rate: float, hz: float=100.0) -> dict` â€” Raw Vicon markers for the player's live 3-D pane: int16 mm at ~hz, -32768 = missing.
- `rs_dir_of(video_dir: Path) -> Path` â€” The RealSense folder an output folder results/video/<P>/<Trial>_<cond> was built from.
- `rebuild(video_dir: Path)` â€” Rebuild an existing player (reuses its MP4s and Vicon CSV, keeps a manual offset), e.g. after adding a pane.
- `build(rs_dir: Path, out: Path | None=None, vicon: Path | None=None, offset: float | None=None, crf: int=23, force: bool=False)` â€” Export (or reuse) the cameras, sync to Vicon, and write player.html with any mesh/EMG panes.
- `main()` â€” (no docstring)

### `scripts/shared/relabel_markers.py`

Label an unlabeled (but complete) Vicon marker cloud — positions untouched.

- `write_wide_csv(path: Path, markers: np.ndarray, labels: list[str])` â€” Write (T, N, 3) mm markers + labels as a single-header wide Nexus CSV.
- `main()` â€” (no docstring)

### `scripts/shared/select_frames_for_review.py`

Select 500 frames across all 66 trials for manual review (Vicon/MATLAB), plus a +/-5 frame window around each, per the sampling plan:

- `trial_dir_name(participant: str, trial_id: str) -> str` â€” (no docstring)
- `load_all_events(overview: pd.DataFrame) -> pd.DataFrame` â€” (no docstring)
- `load_all_gaps(overview: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]` â€” (no docstring)
- `per_frame_missing_count(gaps: pd.DataFrame, n_frames: int) -> np.ndarray` â€” How many markers are simultaneously missing at each frame, built from gap start/end runs (interval accumulation) rather than reloading the raw marker CSV.
- `worst_availability_frames(overview: pd.DataFrame, gaps_by_trial: dict, n_frames_by_trial: dict, top_k: int) -> pd.DataFrame` â€” (no docstring)
- `select_category(candidates: pd.DataFrame, quota: int, taken: set, category: str, reason_col: str | None=None) -> pd.DataFrame` â€” Walk ``candidates`` (already ranked/shuffled by the caller) in order, taking the first ``quota`` frames not already in ``taken`` -- this is the "keep it once, fill the remaining quota from that category" rule: a frame skipped here because another category already claimed it does not shrink this c...
- `build_windows(anchors: pd.DataFrame, n_frames_by_trial: dict) -> pd.DataFrame` â€” +/-WINDOW_RADIUS frames around each anchor, clipped to the trial's valid frame range, deduplicated so an overlap between two nearby anchors' windows is listed once (with both anchors recorded).
- `run(seed: int, out_dir: Path) -> None` â€” (no docstring)
- `write_report(out_dir: Path, anchors: pd.DataFrame, windows: pd.DataFrame) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/select_spotcheck_frames.py`

Select, per trial, the frames for a direct Nexus spot-check of the manual_identity_validation method:

- `run(annotations_csv: Path, out_csv: Path, deviation_threshold_mm: float, seed: int) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/triage_dataset.py`

Triage a directory of Vicon hand recordings before fitting.

- `analyse(path: Path) -> list[dict]` â€” Return one record per hand-side present in the file (≥1).
- `analyse_loaded(source: str, markers, labels, rate) -> list[dict]` â€” Per-hand-side QA records for an already-loaded recording.
- `main()` â€” (no docstring)

### `scripts/shared/update_audit_with_spotchecks.py`

Fold the direct-Nexus spot-check results (spotcheck_frames.m) into results/shared/validation/validation_audit.md: frames reviewed, true errors, artifacts, ambiguous cases, and a short conclusion per trial.

- `find_results_csv(explicit: Path | None) -> Path` â€” (no docstring)
- `df_to_md(df: pd.DataFrame, float_cols: tuple=(), floatfmt: str='{:.1f}') -> str` â€” (no docstring)
- `trial_conclusion(n_reviewed: int, n_true: int, n_artifact: int, n_amb: int) -> str` â€” (no docstring)
- `build_section(manifest: pd.DataFrame, results: pd.DataFrame) -> str` â€” (no docstring)
- `run(manifest_csv: Path, results_csv: Path, audit_md: Path) -> None` â€” (no docstring)
- `main(argv=None)` â€” (no docstring)

### `scripts/shared/video_sync_summary.py`

Collect every results/video/*/*/sync.json into results/video/sync_summary.csv.

- `main()` â€” (no docstring)

## Scripts: scripts/soma

### `scripts/soma/build_library.py`

Build the strategySOMA marker library from a trusted labelled trial.

- `main()` â€” (no docstring)

### `scripts/soma/calibrate_static.py`

Calibrate the SOMA marker library to a participant from their static.

- `base_markers(model, lib: MarkerLibrary, dev: str, n: int=2000) -> np.ndarray` â€” (n, M, 3) mm: real base-subject marker clouds (hand-relative) rebuilt from library frames + their own measured residuals (NaN where absent).
- `write_static_html(model, lib: MarkerLibrary, cloud: np.ndarray, out: dict, path, title: str) -> None` â€” Plotly page of a static fit: the fitted mesh, the static markers, and a line from each marker to its model point.
- `main()` â€” (no docstring)

### `scripts/soma/compare_labels.py`

Label a participant's movement trials with another SOMA model and compare with existing labels, slot by slot.


### `scripts/soma/eval_sample.py`

Score SOMA on the manually reviewed 500-frame sample (all participants).

- `main()` â€” (no docstring)

### `scripts/soma/eval_trial.py`

Label a whole raw trial with the trained SOMA labeller and score it.

- `main()` â€” (no docstring)

### `scripts/soma/fit_trials.py`

Gap-fill each SOMA-labelled trial of a participant, then fit, diagnose and render all of it in 10k-frame batches.

- `sh(args, log)` â€” (no docstring)

### `scripts/soma/flag_frames.py`

Per-frame quality flags for a SOMA-labelled, surface-fitted trial.

- `marker_label_signals(per_frame, voted, conf, label_idx, frames)` â€” Per fitted frame and marker: (conf, disagree, unlabelled) of the raw column that the vote assigned to that label.
- `robust_excess(res)` â€” Per-marker excess over its own median in robust sigmas (MAD * 1.4826).
- `grade(low_conf, disagree, excess, max_joint_deg, step_deg, n_present, min_present, args)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/soma/frame_diagnostics.py`

Per-frame diagnostics of a fit: frames whose pose is not natural or changes too fast.


### `scripts/soma/gap_fill.py`

Gap filling for a labelled marker CSV: spline for short gaps, rigid-body (pattern) fill for longer ones.

- `learned(m, fr)` â€” The filler's prediction of marker ``m`` at frames ``fr`` from the other markers observed there.
- `frame(P)` â€” Origin and orthonormal axes from three points (None if they are nearly collinear).
- `reference(m, tri, start, step, limit)` â€” Nearest frame from ``start`` going ``step`` where the marker and the triplet were all seen: (frame, coords).

### `scripts/soma/gap_fill_check.py`

Gap-fill accuracy: cut artificial gaps where gap_fill.py's output has observed markers, fill them again, compare.


### `scripts/soma/keep_observed.py`

Blank the gap-filled values of a labelled export: keep a value only where the raw Vicon export has a point there.


### `scripts/soma/live_viewer.py`

Live 3-D player (three.js) for a movement fit: rotatable mesh + markers, played in real time.


### `scripts/soma/make_fixture.py`

Cut a small real-data example out of one participant's recordings, for the tutorial. NOT FOR RELEASE by itself: the folder is local, and sharing it needs release authorisation for that participant's data.

- `excerpt(trial: str, p: str, first: int, n: int, out: Path) -> dict` â€” Write frames [first, first + n) of ``p``'s raw ``trial`` to ``out``; return its source record.
- `main()` â€” (no docstring)

### `scripts/soma/make_meshes.py`

Render the fitted-MANO mesh animation for every trial that run_fullset.py has finished fitting: results/soma/mesh/<P>/<P>_<trial>.html (the same naming and viewer as the earlier P10/P12 meshes; scripts/mano/animate_mesh.py does the rendering).

- `capture_rate(csv_path: Path) -> float | None` â€” Marker rate from the 'Trajectories' section header, or None.
- `trial_list(out: Path, participants: list[str])` â€” [(participant, slug, raw_csv, fit_dir)] for every logged trial of these participants.
- `flagged_participants(status: Path) -> set[str]` â€” (no docstring)
- `render(p, sg, raw, tdir, mesh_dir, args) -> bool` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/soma/mesh_video.py`

Render a window of a trial's fitted MANO mesh to MP4 and add it to that trial's RealSense player.

- `fit_dir_of(video_dir: Path) -> Path | None` â€” results/soma/fullset/<P>/<slug> of a video folder (fit slugs spell "Trial 1" as Trial_1).
- `main()` â€” (no docstring)

### `scripts/soma/movement_fit.py`

Fit a movement trial with a final static (static_final_fit.py): shape, scale and attachments held, per-frame pose.

- `fwd(go, hp, tr)` â€” (no docstring)

### `scripts/soma/palm1_pin_test.py`

Does P10's pinned Palm1 vertex hurt? Refit P10 Trial2 hands only three ways, identical settings (P10 config, --cuda-graph --mv-chunk 3000 --mv-overlap 150):

- `main()` â€” (no docstring)

### `scripts/soma/photo_overlay.py`

Overlay a participant's fitted static (mesh + markers) on their photo of the marked hand.

- `focal_px(im: Image.Image) -> float` â€” Focal length in pixels from EXIF: 35 mm-equivalent over the 4:3 sensor diagonal, long side kept if cropped.
- `detect_dots(im: Image.Image) -> list[tuple[float, float, int]]` â€” Centres (x, y, area) of the small grey marker blobs: low saturation, mid brightness.
- `make_camera(f, size)` â€” (no docstring)
- `solve_pose(project, X, x, n_start=150, seed=0)` â€” Rotation + translation (focal fixed) of centred points X to pixels x; best of random starts.
- `auto_label(P3: dict, dots: np.ndarray, px_per_mm: float, tol: float=25.0) -> dict` â€” Rigid marker -> dot index. The rigid markers, seen from the dorsal side, are placed on the dots by a similarity search (mirror, scale, angle, and which marker sits on which dot); the hypothesis with the most markers within ``tol`` px of a dot wins, ties by mean error. False blobs and missing dots...
- `label_fingers(P3, px, project, cam, mean, dots, tol=220.0) -> dict` â€” Finger/thumb markers (which move between the static and the photo): the free dots nearest their projected positions, one dot per marker.
- `fit_camera(P3, px, project)` â€” Camera from the rigid markers; the palm markers' labels are the usual swap, so try all 6 orders.
- `ray_hits(V, F, cam, mean, f, size, pixels) -> list[tuple[float, int, np.ndarray]]` â€” First surface hit of the ray through each pixel: (distance mm along the ray, face, barycentric) or None. Moller-Trumbore against every face; the camera sees the dorsal surface first.
- `nearest_to_ray(V, cam, mean, f, size, pixel) -> tuple[int, float]` â€” Vertex nearest the camera ray through a pixel (for a ray that grazes past the mesh edge) and its distance, mm.
- `skin_mask(im: Image.Image) -> np.ndarray` â€” Pink skin (not the yellower wood or the dark tape) with the marker patches closed.
- `find_webs(im: Image.Image, px: dict, names) -> dict` â€” Pixel position of each finger web: walking from the gap between two fingertip markers toward the gap between their MCP markers, the first skin pixel. Skin mask without the closing that would fill the narrow gaps.
- `palm_rows(mask, px, f, cam, mean, P3) -> dict` â€” Where the hand's edges are along two lines across the palm (through Index1-Pinky1's midpoint, and through Palm2, both parallel to Index1->Pinky1): signed mm from the line's anchor, positive toward Pinky1. Targets for fit_surface_markers --palm-widths. The photo is a flat projection, so this is th...
- `draw(im, project, p, mean, view, P3, px, rigid, rigid_rms)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/soma/pick_window.py`

Pick the stillest, flattest 200 frames of a mapped trial to stand in for a missing static (what was done for P1/P2: ``calibrate_static.py --session ... --window a:b``).

- `window_stats(X: np.ndarray, names: list[str], length: int, stride: int, min_present: float)` â€” [(start, motion_mm_per_frame, bend_deg, min_marker_presence)] over complete-enough windows.
- `choose(stats)` â€” (no docstring)
- `label_agreement(raw_window: np.ndarray, names: list[str], model_path: str, device: str)` â€” (agreement among labelled points, share left unlabelled) of Vicon's column labels vs SOMA's.
- `main()` â€” (no docstring)

### `scripts/soma/preview_labels.py`

Side-by-side 3-D animation: Vicon's labels vs SOMA's, no fit needed.

- `chains(X, L)` â€” Line segments per finger + palm + forearm, as (name, colour, (k,3) with NaN breaks).
- `main()` â€” (no docstring)

### `scripts/soma/preview_synth.py`

Does the synthetic data look like the real data? Numbers + a visual preview.

- `colour(name)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/soma/provenance.py`

Record which version of the fit code a fit started with.

- `snapshot(source: str='driver', started_at: float | None=None, repo: Path=REPO, code=CODE) -> dict` â€” The code state of ``repo`` right now. Hashes are of the files on disk (what a new process imports).
- `write(path: Path, snap: dict) -> None` â€” (no docstring)
- `watch(out: Path, participants: list[str], poll: float=15.0, give_up_min: float=600.0)` â€” (no docstring)
- `reconstruct(out: Path, participants: list[str])` â€” Provenance for fits that started before anyone recorded it, from file times + git history.
- `main()` â€” (no docstring)

### `scripts/soma/quickstart_synth.py`

Synthetic quick-start: a made-up participant generated from MANO alone, then gap-filled and fitted.

- `markers(J, att: torch.Tensor, go, hp, tr, standoff=3.0) -> np.ndarray` â€” Marker positions (T, M, 3) mm: attached vertices of the mean-shape MANO, ``standoff`` mm along the normal.
- `poses(J, T: int, rng, every: int=250, frac: float=0.4) -> np.ndarray` â€” Smooth random hand poses (T, 45): keyframes inside ``frac`` of the joint limits, no twist or PIP/DIP abduction.
- `run(args: list, log: Path)` â€” Run a repo script with this interpreter, output to ``log``; raise if it fails.
- `main()` â€” (no docstring)

### `scripts/soma/run_fullset.py`

Run the SOMA pipeline for many participants: static -> calibrate -> train -> label every trial -> surface fit -> frame flags -> angles CSV.

- `resolve_static(p: str, sessions: dict[str, str], pdir: Path, overrides: dict[str, str] | None=None) -> tuple[Path | None, str]` â€” (static csv, how it was found). Never guesses beyond an exact name match.
- `parse_kv(s: str) -> dict[str, str]` â€” 'P5=1900:2100,P6=47975:48175' -> {'P5': '1900:2100', ...}.
- `make_pseudo_static(raw_csv: Path, window: str, out: Path) -> int` â€” Write frames a:b of a movement trial (Vicon's labels, all columns) as the stand-in static the surface fit reads. Returns the number of frames.
- `resolve_trials(p: str, sessions: dict[str, str], pdir: Path)` â€” [(session, slug, path or None)] for every logged session except the static.
- `fit_args_from_config(cfg_path: Path) -> list[str]` â€” CLI flags for fit_surface_markers.py from a configs/surface_fit/*.json.
- `present_markers(static_csv: Path) -> set[str]` â€” Hand markers the fit can use: seen in at least one frame of the static (the same rule fit_surface_markers.py applies when it builds its marker list).
- `prune_attach(cfg_path: Path, present: set[str], out: Path) -> tuple[Path | None, list[str]]` â€” Copy of the config's attachment map restricted to markers the static has. The fit script refuses an attachment for a marker it is not fitting, and a marker absent from the static cannot be fitted. Returns (file, dropped).
- `parse_calibration(text: str) -> dict` â€” Quality facts from calibrate_static.py's printout.
- `parse_static_fit_mm(text: str) -> float | None` â€” Last 'MEAN |d|' in a fit_surface_markers log (= the stage-2 surface fit).
- `static_verdict(cal: dict, fit_mm: float | None, args) -> list[str]` â€” Reasons the static looks bad (empty = assume good).
- **class `Run`** â€” (no docstring)
  - `.note(p, trial, stage, status, secs=0.0, detail='')` â€” 
  - `.warn(p, reason, detail)` â€” Not blocking: the participant is processed, but the result is weaker.
  - `.flag(p, reason, detail)` â€” 
  - `.sh(cmd, log: Path) -> tuple[int, str, float]` â€” Run ``cmd`` writing its output to ``log``; STALLED if the log stops growing for --stall-min.
- `part(path: Path) -> Path` â€” Temporary name for an output (same extension): renamed only on success.
- `commit(pairs)` â€” (no docstring)
- `step(R: Run, p, trial, stage, outputs, cmd, log, parse=None)` â€” Run ``cmd`` unless every final output exists. Outputs are (tmp, final) pairs already substituted into ``cmd``. Returns (ok, log text or '').
- `write_angles(fit_npz: Path, flags_csv: Path, out_csv: Path)` â€” (no docstring)
- `do_participant(R: Run, p: str, sessions: dict[str, str])` â€” (no docstring)
- `plan(args, sessions_all)` â€” (no docstring)
- `dry_run(args, ps, sessions_all)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/soma/score_observed.py`

Score fit_trials.py runs on the markers that were really observed, so runs with different gap fills compare fairly.


### `scripts/soma/static_final_fit.py`

Final static fit: every marker within 5 mm of its attached vertex, the pose natural.

- `check_limits(J, hinge)` â€” (lo, hi) degrees used to judge a pose: anatomical limits, finger twist 5 deg, PIP/DIP abduction ``hinge``.
- `write_view(path, V, F, M, names, Pk, J21, title)` â€” static.html in fit_surface_markers' layout, so static_search.read_view / Judge.judge read it.
- `main()` â€” (no docstring)

### `scripts/soma/static_passes.py`

Repeated static_final_fit.py passes for one participant, each starting from the best pass so far.

- `sha(path: Path) -> str` â€” sha256 of a file.
- `params_file(start: str, participant: str) -> Path` â€” The params file static_final_fit.py reads for ``--start`` (an output folder or a static_search arm).
- `rank(s: dict) -> tuple` â€” Sort key: valid placement first, then a pose within the limits, then the smallest max distance.
- `main()` â€” (no docstring)

### `scripts/soma/static_pose_check.py`

Is a fitted static pose natural? Joint angles of the static fit against the repo's limits.

- `sources(p: str) -> dict[str, tuple[Path, dict]]` â€” {tag: (static csv, options)}: the static the pipeline used, plus the archived real static where there is one.
- `limit_penalty(B, lo, hi, w: float, dev: str, boost_joints=(6, 12), boost: float=5.0)` â€” pose -> w * sum_j weight_j * (distance outside the joint limits, rad)^2, in the measured axes.
- `check(csv: Path, opts: dict, w_pose: float, base_library: Path, mano_dir: str, twist_tol: float=5.0, limit_w: float=0.0, html: Path | None=None, tag: str='')...)` â€” (no docstring)
- `table()` â€” (no docstring)
- `compare(d: Path)` â€” Fit error with vs without the limit penalty, by finger: does constraining the pose cost the pinky/thumb markers more than the rest (a layout difference the pose was absorbing) or nothing (it was only the pose)?
- `main()` â€” (no docstring)

### `scripts/soma/static_search.py`

Search the static surface fit's settings for the best static per participant.

- `configs(n: int, seed: int, participant: str) -> list[dict]` â€” The production setting first, then ``n`` distinct random draws from GRID.
- `fit_cmd(p: str, static: Path, c: dict, rdir: Path) -> list[str]` â€” (no docstring)
- `read_view(html: Path) -> dict` â€” Mesh, faces, named markers and joint centres from a fit_surface_markers static view.
- `nearest_skin(V, F, P)` â€” Nearest point on the mesh surface for each of P (no outward-side preference, so a marker inside the skin is seen as inside): (points, unit outward face normals). Same projection as the fit script's attachment search.
- **class `Judge`** â€” Everything that needs MANO, loaded once per process.
  - `.palm_dorsal(J)` â€” 
  - `.pose_of(V_mm)` â€” MANO pose that reproduces the shown mesh (betas, global scale, orient, transl, pose fitted to its vertices).
  - `.judge(html: Path) -> dict` â€” 
- `fit_one(J, p, static, ci, c, rdir)` â€” Fit and judge one setting into ``rdir`` (resumable: an existing result is only re-scored if stale).
- `run(participants, n, seed)` â€” (no docstring)
- `best_run(p)` â€” The participant's best-scoring search run (finite score), or None.
- `refine(participants)` â€” Each participant's best search setting again, in every arm of ARMS.
- `attached_vertices(p: str, arm: str='v_scale', names=PALM_PINNED) -> dict` â€” Vertex each named marker was attached to in the all-marker fit ``arm``: of the last round's attached face, the corner nearest the marker (the fit logs the face, not the barycentric point).
- `palm(participants, scratch=False)` â€” Step 1 of palm-first: each participant's best setting + --fit-scale on the palm markers alone, pinned to the all-marker fit's attachments; ``scratch``: pinned instead to the attachments of a fresh palm-only fit (arm palm_scratch), written as palm_pinned_scratch.
- `palm_manual(participants, pins_file, radius, arm='palm_manual', widths=None, w_width=1.0, w_web=1.0, thickness=False, init_arm=None)` â€” Palm-only from scratch with hand-picked vertices (``pins_file``: {marker: vertex}, valid for every participant: MANO's topology is shared) that may each move to a better dorsal vertex within ``radius`` mm.
- `fingers(participants, palm_arm, radius=6.0, arm='fingers', finger_pins=None, overrides=None, start_arm=None, one_by_one=())` â€” Step 2 of palm-first: the markers of every finger on the frozen palm of ``palm_arm`` (its shape, scale, orientation, position and its palm attachments held): finger pose and a length per finger segment are fitted. The photo is not used here: the fingers' pose differs between photo and static. ``s...
- `palm_free(participants, scratch=False)` â€” Palm-only again from the pinned palm fit's pose, with the derived pins dropped: each palm marker re-attaches to the nearest allowed face of that mesh (Palm1 included: nothing is pinned).
- `palm_report(onedrive: Path | None, arm: str='palm_pinned')` â€” Table + views of the palm-only fits, with the full fit's scale beside it.
- `refine_report(onedrive: Path | None)` â€” (no docstring)
- `report(onedrive: Path | None)` â€” (no docstring)
- `main()` â€” (no docstring)

### `scripts/soma/train.py`

Train the strategySOMA labeller on synthetic clouds only.

- `accuracy(net, pts, valid, label, bs=512)` â€” (hard one-to-one acc, best-label argmax acc) over real points (label >= 0).
- `main()` â€” (no docstring)

### `scripts/soma/train_filler.py`

Train the learned marker filler (``strategies/soma/fill.py``) on synthetic clouds, real labelled recordings, or both.


### `scripts/soma/trial_report.py`

Per-trial summary of a run_fullset.py output: how extreme the fitted angles are, how well the fingers were fitted, and which flag signals fired.

- `flexion_b2b(joints: np.ndarray) -> np.ndarray` â€” (T, 21, 3) -> (T, 15) degrees; same definition as export_finger_angles.flexion_angles.
- `fit_log_residuals(log: Path) -> tuple[float, float]` â€” (no docstring)
- `joints_from_fit(fit, model, device, frames, batch=20000) -> np.ndarray` â€” (no docstring)
- `main()` â€” (no docstring)
