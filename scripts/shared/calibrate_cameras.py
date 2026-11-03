"""Camera -> Vicon calibration and aligned-depth export for one trial's RealSense cameras.

Method and results: docs/camera_calibration.md. Per camera, in one pass over its bag:
the depth aligned to colour is written for every exported colour frame, MediaPipe Hands
and the marker-dot detector run on every frame, and a candidate pose is solved wherever
the hand is seen. A fixed camera then gets one pooled pose; a moving (head-mounted) one
keeps a pose per frame with the hand in view.

    python scripts/shared/calibrate_cameras.py results/video/P16/Trial2_HOI
    -> calib.json              per camera: fixed/moving, intrinsics, depth scale, residuals,
                               T_vicon_from_cam (4x4, metres) for a fixed camera
       cam*_poses.csv          moving cameras: T_vicon_from_cam per colour frame (12 values) + residuals
       depth_cam*.mkv          aligned depth in mm, FFV1 16-bit lossless, pts as cam*.mp4
       calib_cam*.png          QA: projected Vicon markers on six frames

Needs the trial's SOMA fit (MANO joints, results/soma/fullset) and the MediaPipe hand
landmarker model (downloaded to models/ on first use). Requires: mediapipe, av, pyrealsense2.
"""
import argparse
import csv
import json
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "mano")]

import animate_mesh as am  # noqa: E402
from realsense_player import bag_intrinsics, read_color_frames, rs_dir_of  # noqa: E402
from vicon2mano.core.camera_calib import (detect_dots, is_moving, pooled_landmark_pose, pose_from_landmarks,  # noqa: E402
                                          project, refine_on_dots, to_matrix)
from vicon2mano.core.dataset import MANO_DIR, soma_fit_dir  # noqa: E402
from vicon2mano.core.video_sync import load_trajectories  # noqa: E402
from vicon2mano.core.viz import marker_bone_pairs, marker_color  # noqa: E402

MODEL = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def landmarker():
    """MediaPipe HandLandmarker (image mode, up to 2 hands)."""
    from mediapipe.tasks.python import BaseOptions, vision
    if not MODEL.exists():
        MODEL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), num_hands=2, min_hand_detection_confidence=0.3))


def hands_in(det, rgb: np.ndarray) -> list[np.ndarray]:
    """Every detected hand's 21 landmarks in pixels."""
    import mediapipe as mp
    h, w = rgb.shape[:2]
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    return [np.array([[q.x * w, q.y * h] for q in lm]) for lm in r.hand_landmarks]


def overlay(tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]], bones, colors, intr: dict, out: Path):
    """Montage (3 per row) of (rgb, pose, Vicon markers m): markers and bones projected with each tile's pose."""
    ims = []
    for rgb, p, V in tiles:
        uv = project(p, V, intr)
        ok = np.isfinite(uv).all(1)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        for a, b in bones:
            if ok[a] and ok[b]:
                dr.line([tuple(uv[a]), tuple(uv[b])], fill=(0, 255, 0), width=1)
        for j in np.flatnonzero(ok):
            dr.ellipse([uv[j, 0] - 3, uv[j, 1] - 3, uv[j, 0] + 3, uv[j, 1] + 3], outline=colors[j], width=1)
        ims.append(im)
    w, h = ims[0].size
    m = Image.new("RGB", (3 * w, (len(ims) + 2) // 3 * h))
    for k, im in enumerate(ims):
        m.paste(im, ((k % 3) * w, (k // 3) * h))
    m.save(out)


def calibrate_camera(name: str, bag: Path, vdir: Path, X: np.ndarray, labels, J: dict, det) -> dict:
    """Depth export + pose(s) for one camera; returns its calib.json entry."""
    intr, scale = bag_intrinsics(bag)
    with open(vdir / f"{name}_frames.csv") as f:
        rows = {int(r["frame_number"]): r for r in csv.DictReader(f)}
    frames, keep_rgb = [], {}
    with av.open(str(vdir / f"depth_{name}.mkv"), "w") as out:
        st = out.add_stream("ffv1", rate=60)
        st.width, st.height, st.pix_fmt = intr["width"], intr["height"], "gray16le"
        st.time_base = st.codec_context.time_base = Fraction(1, 1000)
        for n, ts, rgb, depth in read_color_frames(bag, depth=True):
            if n not in rows:                                   # not exported to cam*.mp4 (duplicate timestamp)
                continue
            raw = np.zeros((intr["height"], intr["width"]), np.uint16) if depth is None else \
                np.clip(np.round(depth * 1000), 0, 65535).astype(np.uint16)          # mm, for both cameras
            fr = av.VideoFrame.from_ndarray(raw, format="gray16le")
            fr.pts = int(rows[n]["pts_ms"])
            out.mux(st.encode(fr))
            vf = int(rows[n]["vicon_frame"]) - 1
            f = {"n": n, "ts": rows[n]["timestamp_ms"], "vf": vf, "hands": [], "pose": None,
                 "grey": rgb[::4, ::4].mean(2) if len(frames) % 5 == 0 else None}
            if vf in J and depth is not None:
                f["hands"] = hands_in(det, rgb)
                cands = [c for c in (pose_from_landmarks(h, depth, J[vf], intr) for h in f["hands"]) if c]
                if cands:
                    f["pose"], f["lm_px"] = min(cands, key=lambda c: c[1])
                    f["dots"] = detect_dots(rgb)
                    if len(keep_rgb) < 120 and len(frames) % 30 == 0:   # QA montage candidates
                        keep_rgb[len(frames)] = rgb
            frames.append(f)
        out.mux(st.encode())
    posed = [k for k, f in enumerate(frames) if f["pose"] is not None]
    V = lambda f: X[f["vf"]][np.isfinite(X[f["vf"]]).all(1)]
    entry = {"bag": bag.name, "intrinsics": intr, "sensor_depth_scale_m": scale, "depth": f"depth_{name}.mkv",
             "depth_units": "mm", "frames": len(frames), "frames_with_hand": len(posed)}
    if len(posed) < 10:
        return {**entry, "status": "too few frames with the hand and a MANO fit"}
    moving = is_moving(np.array([f["grey"] for f in frames if f["grey"] is not None]))
    entry["camera"] = "moving" if moving else "fixed"
    if moving:
        lm, mk, nm = [], [], []
        for k in posed:
            f = frames[k]
            f["pose"], res = refine_on_dots(f["pose"], [(V(f), f["dots"])], intr, gates=(12, 8, 6))
            lm.append(f["lm_px"]); mk += list(res); nm.append(len(res))
        with open(vdir / f"{name}_poses.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame_number", "timestamp_ms", "vicon_frame", "lm_px"] +
                       [f"T{i}{j}" for i in range(3) for j in range(4)])
            for k in posed:
                f = frames[k]
                w.writerow([f["n"], f["ts"], f["vf"] + 1, round(f["lm_px"], 2)] +
                           [round(x, 6) for x in to_matrix(f["pose"])[:3].ravel()])
        entry.update(poses=f"{name}_poses.csv", landmark_px_median=float(np.median(lm)),
                     marker_px_median=float(np.median(mk)) if mk else None, markers_per_frame=float(np.median(nm)))
        tiles = [(keep_rgb[k], frames[k]["pose"], X[frames[k]["vf"]]) for k in sorted(keep_rgb)]
    else:
        (vdir / f"{name}_poses.csv").unlink(missing_ok=True)   # a fixed camera has one pose, in calib.json
        hands = [frames[k]["hands"] for k in posed]
        Jk = [J[frames[k]["vf"]] for k in posed]
        sample = posed[:: max(1, len(posed) // 60)]
        score = lambda p: np.median([min(np.median(np.linalg.norm(h - project(p, J[frames[k]["vf"]], intr), axis=1))
                                         for h in frames[k]["hands"]) for k in sample])
        p = min((frames[k]["pose"] for k in sample), key=score)            # the per-frame pose that fits most frames
        p, lres = pooled_landmark_pose(p, hands, Jk, intr)
        p, mres = refine_on_dots(p, [(V(frames[k]), frames[k]["dots"]) for k in posed], intr)
        entry.update(T_vicon_from_cam=to_matrix(p).round(6).tolist(), landmark_px_median=float(np.median(lres)),
                     marker_px_median=float(np.median(mres)), marker_px_p90=float(np.percentile(mres, 90)),
                     marker_pairs=len(mres))
        tiles = [(keep_rgb[k], p, X[frames[k]["vf"]]) for k in sorted(keep_rgb)]
    tiles = [tiles[i] for i in np.linspace(0, len(tiles) - 1, min(6, len(tiles))).astype(int)]
    overlay(tiles, marker_bone_pairs(labels), [marker_color(n) for n in labels], intr, vdir / f"calib_{name}.png")
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path, help="results/video/<P>/<Trial>_<cond>, already synced")
    a = ap.parse_args()
    vdir = a.video_dir.resolve()
    sync = json.loads((vdir / "sync.json").read_text())
    fit_dir = soma_fit_dir(vdir.parent.name, vdir.name)
    if fit_dir is None or not (fit_dir / "fit.npz").exists():
        sys.exit(f"no SOMA fit for {vdir} (the MANO joints are the correspondences)")
    X, labels, _ = load_trajectories(Path(sync["vicon_csv"]))
    X = X / 1000.0
    fit = np.load(fit_dir / "fit.npz")
    model = am.load_model(str(MANO_DIR), "right", int(fit["hand_pose"].shape[1]))
    vfs = set()
    for f in vdir.glob("cam*_frames.csv"):
        with open(f) as fh:
            vfs |= {int(r["vicon_frame"]) - 1 for r in csv.DictReader(fh)}
    vfs = np.array(sorted(v for v in vfs if 0 <= v < min(len(fit["transl"]), len(X))))
    J = dict(zip(vfs.tolist(), np.concatenate([am.fit_joints21_m(model, fit, c) for c in np.array_split(vfs, 20)])))
    det = landmarker()
    rs_dir = rs_dir_of(vdir)
    calib = {"fit": str(fit_dir), "vicon_csv": sync["vicon_csv"]}
    for i, bag in enumerate(sorted(rs_dir.glob("*.bag")), 1):
        calib[f"cam{i}"] = calibrate_camera(f"cam{i}", bag, vdir, X, labels, J, det)
        print(f"cam{i}:", {k: v for k, v in calib[f"cam{i}"].items() if k not in ("intrinsics", "T_vicon_from_cam")}, flush=True)
    (vdir / "calib.json").write_text(json.dumps(calib, indent=1))


if __name__ == "__main__":
    main()
