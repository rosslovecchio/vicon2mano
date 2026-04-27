# vicon2mano

Fit [MANO](https://mano.is.tue.mpg.de/) hand model parameters to Vicon 3D marker data.

## Overview

Given a sequence of 3D marker positions from a Vicon system (typically 22 markers on the hand), `vicon2mano` recovers the MANO pose (θ) and shape (β) parameters that best explain the observed marker cloud.

**No manual labelling required.** Correspondences between Vicon markers and MANO joints are inferred automatically via:
1. Label-based seeding (if your Vicon labels follow a recognisable naming convention)
2. Hungarian-algorithm assignment on normalised point sets (fallback)

The optimisation follows a two-stage strategy inspired by SMPLify / Total Capture:
- **Stage 1**: optimise shape (β) with a zero pose prior
- **Stage 2**: optimise pose (θ) per frame with temporal smoothness + shape fixed

## Installation

```bash
pip install -e ".[dev]"
```

You must download the MANO model weights separately (requires registration):

```
https://mano.is.tue.mpg.de/
```

Place the unpacked files under `data/mano/` (or pass `--mano-dir`).

## Usage

### CLI

```bash
# from a .c3d file
vicon2mano recording.c3d output.npz --side right

# from a Nexus CSV export
vicon2mano recording.csv output.npz --side right --obj-dir meshes/
```

### Python API

```python
from vicon2mano import MANOFitter, load_c3d
from vicon2mano.fitter import FitConfig
from vicon2mano.export import save_npz

markers, labels, fps = load_c3d("recording.c3d")   # (T, 22, 3) mm

cfg = FitConfig(mano_model_path="data/mano", hand_side="right")
fitter = MANOFitter(cfg)
result = fitter.fit(markers, labels)

# result.joints   → (T, 21, 3)  predicted 3-D joint positions (metres)
# result.betas    → (10,)        hand shape
# result.vertices → (T, 778, 3)  full mesh

save_npz(result, "output.npz")
```

## MANO joint order

| Index | Joint       | Index | Joint      |
|-------|-------------|-------|------------|
| 0     | wrist       | 11    | middle_dip |
| 1     | thumb_mcp   | 12    | middle_tip |
| 2     | thumb_pip   | 13    | ring_mcp   |
| 3     | thumb_dip   | 14    | ring_pip   |
| 4     | thumb_tip   | 15    | ring_dip   |
| 5     | index_mcp   | 16    | ring_tip   |
| 6     | index_pip   | 17    | pinky_mcp  |
| 7     | index_dip   | 18    | pinky_pip  |
| 8     | index_tip   | 19    | pinky_dip  |
| 9     | middle_mcp  | 20    | pinky_tip  |
| 10    | middle_pip  |       |            |

## Vicon marker label hints

If your markers use Vicon Nexus naming, correspondences are auto-detected.
Common patterns recognised (prefix `R`/`L` is stripped):

| Vicon label fragment | MANO joint   |
|----------------------|--------------|
| `WRB`, `WRA`, `WRI`  | wrist        |
| `THB1` … `THB4`      | thumb MCP→tip|
| `IDX1` … `IDX4`      | index MCP→tip|
| `MID1` … `MID4`      | middle MCP→tip|
| `RNG1` … `RNG4`      | ring MCP→tip |
| `LIT1` … `LIT4`      | pinky MCP→tip|

The 22nd marker (often a palm/dorsum marker) is silently ignored if it cannot
be matched within the z-score threshold.

## References

- Romero, J., Tzionas, D., & Black, M. J. (2017). Embodied hands: Modeling and
  capturing hands and bodies together. *ACM TOG (SIGGRAPH Asia)*.
- Bogo, F., et al. (2016). Keep it SMPL. *ECCV*.
- Xiang, D., et al. (2019). Monocular total capture. *CVPR*.
