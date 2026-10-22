#!/usr/bin/env python3
"""Overlay a participant's fitted static (mesh + markers) on their photo of the marked hand.

The photo is a dorsal view of the hand with the markers on. Its small grey dots are found, matched to the
participant's static markers (2-D similarity search in the hand plane, then a rigid-marker camera fit), and the
fitted mesh is projected into the photo with that camera. The camera is fitted ONLY on the markers that do not move
relative to each other (Palm1-3, Thumb1, Index1-Pinky1) with the focal length taken from the photo's EXIF, so how
far the finger markers / mesh fingers land from the photo's is a measurement, not something the fit hid.

    python scripts/soma/photo_overlay.py --participant P4 --arm palm_manual_r0
    python scripts/soma/photo_overlay.py --participant P4 --dots dots.json     # your own {marker: [x, y]} if the automatic labelling is wrong

Photos live in <DATA_ROOT>/soma_transfer/photos/<P>/ (first image there unless --photo). Output: PNG + JSON (camera,
dots, per-marker pixel error) in results/soma/photo_overlay/<P>/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RIGID = ["Palm1", "Palm2", "Palm3", "Thumb1", "Index1", "Middle1", "Ring1", "Pinky1"]
HAND = RIGID + [f"{f}{k}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for k in (2, 3)]
PHOTOS = "soma_transfer/photos"


def focal_px(im: Image.Image) -> float:
    """Focal length in pixels from EXIF: 35 mm-equivalent over the 4:3 sensor diagonal, long side kept if cropped."""
    ex = im.getexif().get_ifd(0x8769)
    f35 = float(ex.get(0xA405) or 24.0)
    return f35 / 43.27 * 1.25 * max(im.size)


def detect_dots(im: Image.Image) -> list[tuple[float, float, int]]:
    """Centres (x, y, area) of the small grey marker blobs: low saturation, mid brightness."""
    a = np.asarray(im.convert("RGB")).astype(float)
    mx, mn = a.max(2), a.min(2)
    grey = ((mx - mn) / np.maximum(mx, 1) < 0.20) & (mx > 80) & (mx < 215)
    lab, n = ndimage.label(grey)
    area = ndimage.sum(grey, lab, range(1, n + 1))
    lo, hi = 1.2e-4 * a.shape[0] * a.shape[1] / 2, 7e-4 * a.shape[0] * a.shape[1]          # 150..1600 px on a 1330x2364 photo
    keep = [i + 1 for i, s in enumerate(area) if lo < s < hi]
    return [(*ndimage.center_of_mass(lab == i)[::-1], int(area[i - 1])) for i in keep]


def make_camera(f, size):
    cx, cy = size[0] / 2, size[1] / 2

    def project(p, X):
        c = X @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[3:6]
        return f * c[:, :2] / c[:, 2:3] + np.array([cx, cy])
    return project


def solve_pose(project, X, x, n_start=150, seed=0):
    """Rotation + translation (focal fixed) of centred points X to pixels x; best of random starts."""
    rng, best = np.random.default_rng(seed), None
    for _ in range(n_start):
        p0 = np.r_[Rotation.random(random_state=int(rng.integers(1 << 30))).as_rotvec(), 0, 0, 450.0]
        r = least_squares(lambda q: (project(q, X) - x).ravel(), p0, loss="soft_l1", f_scale=12,
                          bounds=(np.r_[[-np.inf] * 5, 150], np.r_[[np.inf] * 5, 1500]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x, np.linalg.norm(project(best.x, X) - x, axis=1)


def auto_label(P3: dict, dots: np.ndarray, px_per_mm: float, tol: float = 25.0) -> dict:
    """Rigid marker -> dot index. The rigid markers, seen from the dorsal side, are placed on the dots by a
    similarity search (mirror, scale, angle, and which marker sits on which dot); the hypothesis with the most
    markers within ``tol`` px of a dot wins, ties by mean error. False blobs and missing dots are tolerated."""
    names = [n for n in RIGID if n in P3]
    X = np.array([P3[n] for n in names])
    n = np.cross(P3["Palm2"] - P3["Palm1"], P3["Palm3"] - P3["Palm1"])
    n /= np.linalg.norm(n)
    e1 = P3["Index1"] - P3["Pinky1"]
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    Q = np.c_[(X - X.mean(0)) @ e1, (X - X.mean(0)) @ np.cross(n, e1)]
    best = (-1, np.inf, None)
    for mirror, s, th in itertools.product((1, -1), px_per_mm * np.array([.6, .7, .8, .9, 1, 1.15, 1.3, 1.5]), np.radians(np.arange(0, 360, 3))):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        M = s * (Q * [1, mirror]) @ R.T
        for i in range(len(names)):
            T = dots - M[i]                                          # each dot as the position of marker i
            D = np.linalg.norm((M[None] + T[:, None])[:, :, None] - dots[None, None], axis=3).min(2)   # (dots, markers)
            hit = (D < tol).sum(1)
            j = int(np.argmax(hit - D.mean(1) * 1e-3))
            if (hit[j], -D[j].mean()) > (best[0], -best[1]):
                best = (int(hit[j]), float(D[j].mean()), M + T[j])
    cost = np.linalg.norm(best[2][:, None] - dots[None], axis=2)
    r, c = linear_sum_assignment(cost)
    return {names[i]: int(j) for i, j in zip(r, c) if cost[i, j] < tol}


def label_fingers(P3, px, project, cam, mean, dots, tol=220.0) -> dict:
    """Finger/thumb markers (which move between the static and the photo): the free dots nearest their projected
    positions, one dot per marker."""
    names = [n for n in HAND if n not in px and n in P3]
    free = [j for j in range(len(dots)) if not any(np.allclose(dots[j], v) for v in px.values())]
    pred = project(cam, np.array([P3[n] for n in names]) - mean)
    cost = np.linalg.norm(pred[:, None] - dots[free][None], axis=2)
    r, c = linear_sum_assignment(cost)
    return {names[i]: dots[free[j]] for i, j in zip(r, c) if cost[i, j] < tol}


def fit_camera(P3, px, project):
    """Camera from the rigid markers; the palm markers' labels are the usual swap, so try all 6 orders."""
    best = None
    for perm in itertools.permutations(["Palm1", "Palm2", "Palm3"]):
        lab = {**{k: px[k] for k in RIGID[3:] if k in px}, **{a: px[b] for a, b in zip(perm, ["Palm1", "Palm2", "Palm3"]) if b in px}}
        X = np.array([P3[k] for k in lab]); mean = X.mean(0)
        p, e = solve_pose(project, X - mean, np.array([lab[k] for k in lab]), n_start=60)
        rms = float(np.sqrt((e ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, p, mean, perm, dict(zip(lab, e)))
    return best


def draw(im, project, p, mean, view, P3, px, rigid, rigid_rms):
    ov = Image.new("RGBA", im.size)
    d = ImageDraw.Draw(ov)
    V2 = project(p, view["V"] - mean)
    for f in view["F"]:
        pts = [tuple(V2[i]) for i in f]
        d.line(pts + [pts[0]], fill=(0, 200, 255, 90), width=1)
    for n in px:                                          # the static's markers, from the static (the mesh view may lack some)
        c = project(p, (P3[n] - mean)[None])[0]
        q, col = px[n], (255, 0, 0, 255) if n in rigid else (255, 140, 0, 255)
        d.line([tuple(q), tuple(c)], fill=col, width=2)
        d.ellipse([c[0] - 7, c[1] - 7, c[0] + 7, c[1] + 7], outline=col, width=3)
        d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=(0, 255, 0, 255))
        d.text((c[0] + 9, c[1] - 14), n, fill=(255, 255, 255, 255))
    d.text((10, 10), f"camera fit on rigid markers: {rigid_rms:.1f} px RMS", fill=(255, 255, 255, 255))
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def main():
    import static_search as ss
    import static_pose_check as spc
    from vicon2mano.core.dataset import DATA_ROOT
    from vicon2mano.core.loader import load_csv
    from vicon2mano.strategies.soma.calibrate import static_cloud

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", required=True)
    ap.add_argument("--arm", default="palm_manual_r0", help="static_search arm whose static.html is overlaid")
    ap.add_argument("--photo", default=None)
    ap.add_argument("--dots", default=None, help="JSON {marker: [x, y]} (pixels) instead of the automatic labelling")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = a.participant

    photo = Path(a.photo) if a.photo else next(iter(sorted((Path(DATA_ROOT) / PHOTOS / p).glob("*.[jJpP][pPnN][gGeE]*"))), None)
    if photo is None:
        raise SystemExit(f"no photo for {p}: put one in {Path(DATA_ROOT) / PHOTOS / p}")
    im = Image.open(photo)
    X, names = load_csv(str(spc.sources(p)[p][0]))[:2]
    P3 = dict(zip([n.split(":")[-1] for n in names], static_cloud(X, 30)[0]))
    f = focal_px(im)
    project = make_camera(f, im.size)

    if a.dots:
        px = {k: np.array(v, float) for k, v in json.loads(Path(a.dots).read_text()).items()}
    else:
        dots = np.array([d[:2] for d in detect_dots(im)])
        print(f"{len(dots)} dots found; focal {f:.0f} px")
        px = {k: dots[j] for k, j in auto_label(P3, dots, f / 300.0).items()}   # ~30 cm camera distance to search around
    if sum(k in px for k in RIGID) < 6:
        raise SystemExit(f"only {sum(k in px for k in RIGID)} rigid markers matched to dots: pass --dots")
    rms, cam, mean, perm, err = fit_camera(P3, px, project)
    px = {**{k: v for k, v in px.items() if not k.startswith("Palm")}, **dict(zip(perm, [px[k] for k in ("Palm1", "Palm2", "Palm3") if k in px]))}
    if not a.dots:
        px |= label_fingers(P3, px, project, cam, mean, dots)
    print(f"camera fit on {len(err)} rigid markers: {rms:.1f} px RMS (about {rms * np.linalg.norm(cam[3:6]) / f:.1f} mm)"
          + ("   <-- LARGE: check the labelling (--dots)" if rms > 8 else ""))
    far = {k: float(np.linalg.norm(project(cam, (P3[k] - mean)[None])[0] - px[k])) for k in HAND if k in px and k not in err}
    print("markers outside the camera fit (fingers, thumb), static vs photo (px):", {k: round(v) for k, v in far.items()})

    view = ss.read_view(ss.OUT / p / a.arm / "static.html")
    out = Path(a.out) if a.out else REPO / "results/soma/photo_overlay" / p / f"{p}_photo_overlay_{a.arm}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(im, project, cam, mean, view, P3, px, err, rms).save(out)
    out.with_suffix(".json").write_text(json.dumps(dict(
        photo=str(photo), focal_px=f, rigid_rms_px=rms, camera=[float(v) for v in cam], mean=[float(v) for v in mean],
        dots={k: [float(v[0]), float(v[1])] for k, v in px.items()}, finger_error_px=far), indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
