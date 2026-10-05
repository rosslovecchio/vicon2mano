#!/usr/bin/env python3
"""Fold the direct-Nexus spot-check results (spotcheck_frames.m) into
results/shared/validation/validation_audit.md: frames reviewed, true
errors, artifacts, ambiguous cases, and a short conclusion per trial.

Idempotent: re-running (e.g. after more cases get reviewed) replaces the
previously-appended spot-check section rather than duplicating it, so this
can be run again and again as the review progresses.

Usage
-----
python scripts/shared/update_audit_with_spotchecks.py
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

MANIFEST_CSV = REPO_ROOT / "results" / "shared" / "validation" / "spotcheck_manifest.csv"
RESULTS_CANDIDATES = [
    REPO_ROOT / "results" / "shared" / "validation" / "spotcheck_results.csv",
    Path(r"C:\Users\RL000009\MATLAB\Projects\VICON_Labelling\gesture_labelling\spotcheck_results.csv"),
]
AUDIT_MD = REPO_ROOT / "results" / "shared" / "validation" / "validation_audit.md"

SECTION_MARKER = "## Direct Nexus spot-check review"
VERDICTS = ("true_vicon_error", "validation_artifact", "confirmed_ambiguous")


def find_results_csv(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"{explicit} not found")
        return explicit
    for cand in RESULTS_CANDIDATES:
        if cand.exists():
            return cand
    raise SystemExit(
        "spotcheck_results.csv not found in any default location:\n  "
        + "\n  ".join(str(c) for c in RESULTS_CANDIDATES)
        + "\nReview at least one case with spotcheck_frames.m first.")


def df_to_md(df: pd.DataFrame, float_cols: tuple = (), floatfmt: str = "{:.1f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols and not pd.isna(v):
                cells.append(floatfmt.format(v))
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def trial_conclusion(n_reviewed: int, n_true: int, n_artifact: int, n_amb: int) -> str:
    if n_reviewed == 0:
        return "not yet reviewed"
    if n_true > n_artifact and n_true > n_amb:
        return f"likely real labelling errors ({n_true}/{n_reviewed} confirmed true_vicon_error)"
    if n_artifact > n_true and n_artifact > n_amb:
        return f"likely methodological ({n_artifact}/{n_reviewed} validation_artifact -- distrust this trial's wrong_label/ghost verdicts until re-checked)"
    if n_amb >= n_reviewed:
        return f"genuinely hard to call ({n_amb}/{n_reviewed} confirmed_ambiguous)"
    return f"mixed ({n_true} true / {n_artifact} artifact / {n_amb} ambiguous of {n_reviewed} reviewed) -- no single explanation dominates"


def build_section(manifest: pd.DataFrame, results: pd.DataFrame) -> str:
    merged = manifest.merge(
        results[["participant", "trial_id", "frame", "marker_name", "verdict", "notes"]],
        on=["participant", "trial_id", "frame", "marker_name"], how="left")

    n_selected = len(manifest)
    n_reviewed = int(merged["verdict"].notna().sum())
    overall_counts = {v: int((merged["verdict"] == v).sum()) for v in VERDICTS}

    pct_reviewed = 100 * n_reviewed / n_selected if n_selected else 0.0

    L = [SECTION_MARKER + "\n"]
    L.append(f"Frames selected for spot-check: {n_selected}. "
             f"Reviewed so far: {n_reviewed} ({pct_reviewed:.0f}%).")
    L.append(f"\n- `true_vicon_error`: {overall_counts['true_vicon_error']}")
    L.append(f"- `validation_artifact`: {overall_counts['validation_artifact']}")
    L.append(f"- `confirmed_ambiguous`: {overall_counts['confirmed_ambiguous']}\n")

    rows = []
    for (p, t), g in merged.groupby(["participant", "trial_id"]):
        n_sel = len(g)
        n_rev = int(g["verdict"].notna().sum())
        n_true = int((g["verdict"] == "true_vicon_error").sum())
        n_art = int((g["verdict"] == "validation_artifact").sum())
        n_amb = int((g["verdict"] == "confirmed_ambiguous").sum())
        rows.append({
            "participant": p, "trial_id": t, "n_selected": n_sel, "n_reviewed": n_rev,
            "true_errors": n_true, "artifacts": n_art, "ambiguous": n_amb,
            "conclusion": trial_conclusion(n_rev, n_true, n_art, n_amb),
        })
    per_trial = pd.DataFrame(rows).sort_values(["participant", "trial_id"])

    L.append("### Per-trial results\n")
    cols = ["participant", "trial_id", "n_selected", "n_reviewed", "true_errors",
            "artifacts", "ambiguous", "conclusion"]
    L.append(df_to_md(per_trial[cols]))
    L.append("")

    notes = merged[merged["notes"].notna() & (merged["notes"].astype(str).str.len() > 0)]
    if len(notes):
        L.append("### Reviewer notes\n")
        for _, r in notes.iterrows():
            L.append(f"- {r['participant']}/{r['trial_id']} frame {r['frame']} "
                     f"({r['marker_name']}, {r['verdict']}): {r['notes']}")
        L.append("")

    return "\n".join(L)


def run(manifest_csv: Path, results_csv: Path, audit_md: Path) -> None:
    manifest = pd.read_csv(manifest_csv)
    results = pd.read_csv(results_csv)

    section = build_section(manifest, results)

    if audit_md.exists():
        existing = audit_md.read_text(encoding="utf-8")
        marker_pos = existing.find(SECTION_MARKER)
        base = existing[:marker_pos].rstrip() if marker_pos != -1 else existing.rstrip()
    else:
        base = f"# Manual identity validation: audit report\n\n(no prior audit found at {audit_md})"

    audit_md.write_text(base + "\n\n" + section + "\n", encoding="utf-8")
    print(f"Updated {audit_md} with spot-check results "
          f"({int(results.shape[0])} reviewed case(s) in spotcheck_results.csv)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest-csv", default=str(MANIFEST_CSV))
    ap.add_argument("--results-csv", default=None)
    ap.add_argument("--audit-md", default=str(AUDIT_MD))
    args = ap.parse_args(argv)

    results_csv = find_results_csv(Path(args.results_csv) if args.results_csv else None)
    run(Path(args.manifest_csv), results_csv, Path(args.audit_md))


if __name__ == "__main__":
    main()
