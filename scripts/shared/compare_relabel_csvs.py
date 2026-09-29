#!/usr/bin/env python3
"""Compare a raw Vicon export against manually-relabelled version(s) of it.

Built for the P10/Trial2 Hands-only triple:

    Trial2_handsonly.csv                       -- raw Nexus export
    Trial2_handsonly_manuallylabelled.csv      -- manual relabel, gaps kept
    Trial2_handsonly_manuallylabelled_filled.csv -- manual relabel, gap-filled

but works for any 2-3 CSVs sharing the same marker set and frame count.

For each non-raw file it reports, per marker:
  - NaN rate (missing frames)
  - fraction of present frames whose position changed vs raw (beyond
    --move-tol, so float round-trip noise doesn't count as a "correction")
  - median/p95 displacement on frames that did change
  - whether the change looks like a *relabel* -- i.e. the new position at
    frame t matches (within tolerance) some *other* raw marker's position at
    frame t -- vs a same-label *fill*/*correction* that doesn't match any
    other marker

Also reports, per anatomically-adjacent bone (`infer_bones` — finger segments
plus palm-plate pairs, the same topology the cascade strategy uses):
  - mean length and std (rigidity) in raw vs. each comparison file. A relabel
    that fixes a real mislabel should *shrink* std (bone length is close to
    constant for a correctly labelled rigid/near-rigid segment); a relabel
    that's wrong, or a bad fill, tends to grow it or shift the mean.

Plots (saved as PNG under --out-dir):
  - per-marker relabel-rate bar chart
  - frame x marker heatmap of which (frame, marker) cells changed
  - displacement histogram for changed cells
  - missing-data-over-time, raw vs each comparison file
  - per-bone length std, raw vs each comparison file (rigidity check)
  - per-bone length distributions (box plot), raw vs each comparison file

Usage
-----
python scripts/shared/compare_relabel_csvs.py \
    --raw ".../P10/Trial2_handsonly.csv" \
    --compare ".../P10/Trial2_handsonly_manuallylabelled.csv" \
    --compare ".../P10/Trial2_handsonly_manuallylabelled_filled.csv" \
    --out-dir results/mano/p10_trial2_relabel_check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vicon2mano.core.loader import load_csv
from vicon2mano.core.viz import marker_color
from vicon2mano.strategies.cascade.quality_cascade import infer_bones

MOVE_TOL_MM = 1.0    # below this, treat as "unchanged" (export round-trip noise)
SWAP_TOL_MM = 5.0    # within this of another raw marker -> looks like a relabel


def short_label(label: str) -> str:
    return label.split(":")[-1]


def load_all(raw_path: Path, compare_paths: list[Path]):
    raw_m, raw_labels = load_csv(str(raw_path))
    datasets = [(raw_path.name, raw_m, raw_labels)]
    for p in compare_paths:
        m, labels = load_csv(str(p))
        if labels != raw_labels:
            raise ValueError(
                f"{p.name}: marker set differs from raw ({labels} != {raw_labels})"
            )
        if m.shape[0] != raw_m.shape[0]:
            raise ValueError(
                f"{p.name}: frame count {m.shape[0]} != raw {raw_m.shape[0]}"
            )
        datasets.append((p.name, m, labels))
    return datasets


def classify_changes(raw: np.ndarray, cmp: np.ndarray) -> dict:
    """Per-marker stats for `cmp` vs `raw`. raw, cmp: (T, N, 3)."""
    T, N, _ = raw.shape
    raw_present = np.isfinite(raw).all(axis=2)      # (T, N)
    cmp_present = np.isfinite(cmp).all(axis=2)      # (T, N)

    dist = np.full((T, N), np.nan, dtype=np.float64)
    both = raw_present & cmp_present
    dist[both] = np.linalg.norm((cmp - raw)[both], axis=-1)

    changed = both & (dist > MOVE_TOL_MM)   # (T, N)

    # For changed cells, check whether the new position matches some *other*
    # raw marker at that frame (a relabel), rather than the same marker just
    # moving to a corrected-but-nearby position (a fill/jitter fix).
    is_relabel = np.zeros((T, N), dtype=bool)
    ft, fm = np.nonzero(changed)
    for t, m in zip(ft, fm):
        others = raw[t]  # (N, 3)
        d = np.linalg.norm(others - cmp[t, m], axis=-1)
        d[m] = np.inf
        if np.nanmin(d) < SWAP_TOL_MM:
            is_relabel[t, m] = True

    return {
        "raw_present": raw_present,
        "cmp_present": cmp_present,
        "dist": dist,
        "changed": changed,
        "is_relabel": is_relabel,
    }


def bone_lengths(markers: np.ndarray, bones: list[tuple[int, int, str]]) -> np.ndarray:
    """(T, n_bones) length of each bone per frame; NaN where either end is missing."""
    i_idx = [i for i, _j, _n in bones]
    j_idx = [j for _i, j, _n in bones]
    return np.linalg.norm(markers[:, i_idx] - markers[:, j_idx], axis=-1)


def print_bone_report(name: str, bone_names: list[str],
                       raw_len: np.ndarray, cmp_len: np.ndarray) -> None:
    print(f"\n--- bone lengths: {name} vs raw ---")
    print(f"{'bone':<28} {'raw mean':>9} {'raw std':>8} "
          f"{'cmp mean':>9} {'cmp std':>8} {'std delta':>10}")
    std_deltas = []
    for k, bname in enumerate(bone_names):
        r, c = raw_len[:, k], cmp_len[:, k]
        r_mean, r_std = np.nanmean(r), np.nanstd(r)
        c_mean, c_std = np.nanmean(c), np.nanstd(c)
        delta = c_std - r_std
        std_deltas.append(delta)
        print(f"{bname:<28} {r_mean:>9.2f} {r_std:>8.2f} "
              f"{c_mean:>9.2f} {c_std:>8.2f} {delta:>+10.2f}")
    std_deltas = np.array(std_deltas)
    improved = (std_deltas < -0.5).sum()
    worsened = (std_deltas > 0.5).sum()
    print(f"\n{improved}/{len(bone_names)} bones got more rigid (std -0.5mm+), "
          f"{worsened}/{len(bone_names)} got less rigid (std +0.5mm+)")


def make_bone_plots(name: str, bone_names: list[str],
                     raw_len: np.ndarray, cmp_len: np.ndarray, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name.replace(".csv", "").replace(" ", "_")
    n = len(bone_names)

    raw_std = np.nanstd(raw_len, axis=0)
    cmp_std = np.nanstd(cmp_len, axis=0)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), 4))
    w = 0.4
    ax.bar(x - w / 2, raw_std, width=w, label="raw", color="#95a5a6")
    ax.bar(x + w / 2, cmp_std, width=w, label=name, color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(bone_names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("bone length std (mm)")
    ax.set_title(f"{name}: bone-length rigidity (lower = more consistent)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_bone_std.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 4))
    positions = np.arange(n) * 3
    raw_data = [raw_len[np.isfinite(raw_len[:, k]), k] for k in range(n)]
    cmp_data = [cmp_len[np.isfinite(cmp_len[:, k]), k] for k in range(n)]
    bp1 = ax.boxplot(raw_data, positions=positions - 0.5, widths=0.8,
                      showfliers=False, patch_artist=True)
    bp2 = ax.boxplot(cmp_data, positions=positions + 0.5, widths=0.8,
                      showfliers=False, patch_artist=True)
    for b in bp1["boxes"]:
        b.set_facecolor("#95a5a6")
    for b in bp2["boxes"]:
        b.set_facecolor("#2980b9")
    ax.set_xticks(positions)
    ax.set_xticklabels(bone_names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("bone length (mm)")
    ax.set_title(f"{name}: bone-length distributions (grey=raw, blue={name})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_bone_boxplot.png", dpi=150)
    plt.close(fig)


def print_report(name: str, labels: list[str], stats: dict) -> None:
    T, N = stats["changed"].shape
    print(f"\n=== {name} vs raw ===")
    print(f"{'marker':<16} {'raw NaN%':>9} {'cmp NaN%':>9} "
          f"{'changed%':>9} {'relabel%':>9} {'med mm':>8} {'p95 mm':>8}")
    for i, label in enumerate(labels):
        raw_nan = 100 * (~stats["raw_present"][:, i]).mean()
        cmp_nan = 100 * (~stats["cmp_present"][:, i]).mean()
        changed_frac = 100 * stats["changed"][:, i].mean()
        relabel_frac = 100 * stats["is_relabel"][:, i].sum() / max(
            stats["changed"][:, i].sum(), 1
        )
        d = stats["dist"][stats["changed"][:, i], i]
        med = np.median(d) if d.size else float("nan")
        p95 = np.percentile(d, 95) if d.size else float("nan")
        print(f"{short_label(label):<16} {raw_nan:>8.2f}% {cmp_nan:>8.2f}% "
              f"{changed_frac:>8.2f}% {relabel_frac:>8.2f}% {med:>8.2f} {p95:>8.2f}")

    total_changed = stats["changed"].sum()
    total_relabel = stats["is_relabel"].sum()
    total_present_both = (stats["raw_present"] & stats["cmp_present"]).sum()
    print(f"\nOverall: {total_changed}/{total_present_both} present-cells changed "
          f"({100 * total_changed / max(total_present_both, 1):.2f}%); "
          f"{total_relabel} of those look like relabels "
          f"({100 * total_relabel / max(total_changed, 1):.1f}%)")


def make_plots(name: str, labels: list[str], stats: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name.replace(".csv", "").replace(" ", "_")
    short = [short_label(l) for l in labels]
    colors = [marker_color(l) for l in labels]
    N = len(labels)

    # 1. per-marker relabel-rate bar chart
    changed_pct = 100 * stats["changed"].mean(axis=0)
    relabel_pct = 100 * stats["is_relabel"].mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(8, N * 0.5), 4))
    x = np.arange(N)
    ax.bar(x, changed_pct, color=colors, alpha=0.4, label="changed (any)")
    ax.bar(x, relabel_pct, color=colors, label="looks like relabel")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=60, ha="right")
    ax.set_ylabel("% of frames")
    ax.set_title(f"{name}: per-marker change rate vs raw")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_change_rate.png", dpi=150)
    plt.close(fig)

    # 2. frame x marker heatmap of changes (downsampled if long)
    changed = stats["changed"].T.astype(float)  # (N, T)
    changed[stats["is_relabel"].T] = 2.0
    T = changed.shape[1]
    step = max(1, T // 4000)
    fig, ax = plt.subplots(figsize=(12, max(3, N * 0.3)))
    im = ax.imshow(
        changed[:, ::step], aspect="auto", cmap="viridis", interpolation="nearest",
        extent=(0, T, N, 0),
    )
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("frame")
    ax.set_title(f"{name}: changed cells (bright=relabel-like) vs raw")
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2], label="0=same 1=moved 2=relabel-like")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_change_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. displacement histogram for changed cells
    d = stats["dist"][stats["changed"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    if d.size:
        ax.hist(np.clip(d, 0, np.percentile(d, 99)), bins=60, color="#2980b9")
    ax.set_xlabel("displacement (mm), clipped at p99")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: displacement on changed cells")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_displacement_hist.png", dpi=150)
    plt.close(fig)


def make_missing_plot(datasets, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, m, _labels in datasets:
        present_frac = np.isfinite(m).all(axis=2).mean(axis=1)
        missing_pct = 100 * (1 - present_frac)
        T = len(missing_pct)
        step = max(1, T // 4000)
        ax.plot(np.arange(0, T, step), missing_pct[::step], label=name, lw=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("% markers missing")
    ax.set_title("Missing markers over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "missing_over_time.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--compare", required=True, type=Path, action="append",
                     help="repeatable; each is compared against --raw")
    ap.add_argument("--out-dir", type=Path, default=Path("results/mano/relabel_check"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    datasets = load_all(args.raw, args.compare)
    raw_name, raw_m, raw_labels = datasets[0]

    bones = infer_bones(raw_labels)
    # infer_bones names use full labels ("P10:Thumb1-P10:Thumb2"); shorten
    bone_names = [
        f"{short_label(li)}-{short_label(lj)}"
        for i, j, _n in bones
        for li, lj in [(raw_labels[i], raw_labels[j])]
    ]
    raw_bone_len = bone_lengths(raw_m, bones)

    for name, m, labels in datasets[1:]:
        stats = classify_changes(raw_m, m)
        print_report(name, labels, stats)
        if not args.no_plots:
            make_plots(name, labels, stats, args.out_dir)

        cmp_bone_len = bone_lengths(m, bones)
        print_bone_report(name, bone_names, raw_bone_len, cmp_bone_len)
        if not args.no_plots:
            make_bone_plots(name, bone_names, raw_bone_len, cmp_bone_len, args.out_dir)

    if not args.no_plots:
        make_missing_plot(datasets, args.out_dir)
        print(f"\nPlots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
