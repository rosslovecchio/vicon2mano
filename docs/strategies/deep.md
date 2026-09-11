# Strategy: deep (CNN) marker labeler

*See `../../CLAUDE.md` for the index.*

3-D U-Net that voxelises the raw marker cloud and regresses one heatmap per
MANO joint. Unlike every other strategy here it is fully order-agnostic --
it needs no anchor markers and no existing labels.

- Model: `vicon2mano/strategies/deep/labeler.py` (`MarkerLabeler`, `DeepLabeler`)
- Data: `synth_data.py` (synthetic generation + HDF5 dataset), `real_data.py`
- Scripts: `scripts/deep/train_labeler.py`, `scripts/deep/eval_labeler.py`
- Tests: `tests/deep/`

| Constant | Value | Meaning |
|---|---|---|
| `VOXEL_RES` | 64 | Grid side length (64 cubed) |
| `SPATIAL_HALF` | 0.22 m | Grid half-extent |
| `GAUSSIAN_SIGMA_VOXELS` | 1.5 | Blob std for heatmap targets |
| `CONF_THRESHOLD` | 0.10 | Min sigmoid confidence to accept a match |

Tests must import `VOXEL_RES` rather than hardcoding the grid size.

```bash
python scripts/deep/train_labeler.py generate     --mano-dir ../clean_kinematics/mano_v1_2/models     --n-samples 500000 --out data/synth_labeler.h5

python scripts/deep/train_labeler.py train     --data data/synth_labeler.h5     --out-weights vicon2mano/weights/deep_labeler.pt     --epochs 20 --batch-size 2 --lr 3e-4
```

**Warning:** the checked-in `vicon2mano/weights/deep_labeler.pt` is a
smoke-test checkpoint (trained with `max_samples=500` on one real
recording); its predictions are unreliable. The label-seed path is what
makes real recordings work. Retrain with the full synthetic pipeline before
trusting the CNN.
