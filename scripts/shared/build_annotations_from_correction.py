#!/usr/bin/env python3
"""Build ``manual_annotations.csv`` (deliverable #2) by diffing each
trial's original Vicon-labelled CSV against a hand-corrected export you
made by relabelling directly in Nexus -- the alternative to typing
per-marker verdicts into ``label_selected_frames.m``.

Requires ``visited_frames.csv``, logged by the simplified
``label_selected_frames.m`` (just confirms which of the 500 sampled frames
you actually looked at -- an unedited marker is otherwise indistinguishable
from one you never reviewed).

For each trial with visited frames, the corrected export is discovered
next to the original using the same convention as
``vicon2mano.core.dataset.find_manual_csv``
(``<stem>_manuallylabelled[_filled].csv``) -- export your Nexus correction
under that name.

A marker whose label disappeared during your review (present in the
original, no coordinate in your corrected export) is recorded as
``ghost`` -- per instruction, a label that you deleted during review was
judged not correct, not treated as an ordinary data gap. A coordinate you
*added* where the original had none is gap-filling, out of scope here, and
is skipped with a warning rather than guessed at.

Usage
-----
python scripts/shared/build_annotations_from_correction.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vicon2mano.core.loader import load_csv                       # noqa: E402
from vicon2mano.core import dataset as ds                         # noqa: E402
from vicon2mano.core.agreement import match_markers                # noqa: E402
from vicon2mano.core import validation as val                     # noqa: E402

VISITED_CSV_CANDIDATES = [
    REPO_ROOT / "results" / "shared" / "frame_sample" / "visited_frames.csv",
    Path(r"C:\Users\RL000009\MATLAB\Projects\VICON_Labelling\gesture_labelling\visited_frames.csv"),
]
OUT_CSV = REPO_ROOT / "results" / "shared" / "frame_sample" / "manual_annotations.csv"


def find_visited_csv(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"{explicit} not found")
        return explicit
    for cand in VISITED_CSV_CANDIDATES:
        if cand.exists():
            return cand
    raise SystemExit(
        "visited_frames.csv not found in any default location:\n  "
        + "\n  ".join(str(c) for c in VISITED_CSV_CANDIDATES)
        + "\nRun label_selected_frames.m (visit-logging mode) first, or pass --visited-csv.")


def run(visited_csv: Path, reviewer_override: str | None, out_csv: Path) -> None:
    visited = pd.read_csv(visited_csv)
    if visited.empty:
        raise SystemExit(f"{visited_csv} has no rows yet -- nothing to build annotations from.")

    all_rows = []
    for (participant, trial_id), g in visited.groupby(["participant", "trial_id"]):
        trial_path, *_ = ds.find_trial_csv(participant, trial_id)
        if trial_path is None:
            print(f"[warn] no CSV mapped to {participant}/{trial_id} -- skipping "
                  f"{len(g)} visited frame(s)")
            continue
        corrected_path = ds.find_manual_csv(trial_path)
        if corrected_path is None:
            print(f"[warn] no corrected export found next to {trial_path.name} for "
                  f"{participant}/{trial_id} (expected <stem>_manuallylabelled[_filled].csv) "
                  f"-- skipping {len(g)} visited frame(s)")
            continue

        print(f"{participant}/{trial_id}: {trial_path.name} vs {corrected_path.name}, "
              f"{len(g)} visited frame(s)")
        markers_orig, labels_orig = load_csv(str(trial_path))
        markers_corrected, labels_corrected = load_csv(str(corrected_path))

        reviewer = reviewer_override or (g["reviewer"].iloc[0] if "reviewer" in g else "unknown")
        annotations = val.build_annotations_from_correction(
            markers_orig, labels_orig, markers_corrected, labels_corrected,
            visited_frames=g["frame"].astype(int).tolist(),
            participant=participant, trial_id=trial_id, reviewer=reviewer,
            match_markers_fn=match_markers)
        all_rows.append(annotations)

    if not all_rows:
        raise SystemExit("No trials produced any annotations -- see warnings above.")

    out = pd.concat(all_rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nWrote {len(out)} annotation row(s) covering "
          f"{out.groupby(['participant', 'trial_id']).ngroups} trial(s) to {out_csv}")
    print(out["status"].value_counts().to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--visited-csv", default=None)
    ap.add_argument("--reviewer", default=None,
                    help="override the reviewer name recorded in the output "
                         "(default: use visited_frames.csv's own reviewer column)")
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    visited_csv = find_visited_csv(Path(args.visited_csv) if args.visited_csv else None)
    run(visited_csv, args.reviewer, Path(args.out_csv))


if __name__ == "__main__":
    main()
