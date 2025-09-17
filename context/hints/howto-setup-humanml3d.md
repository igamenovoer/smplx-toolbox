# How to Set Up HumanML3D for FlowMDM Generation

This guide shows how to wire the HumanML3D dataset normalization stats (Mean/Std) into FlowMDM so the HumanML3D sampling path works end‑to‑end in this workspace.

## What FlowMDM Needs
- A HumanML‑trained checkpoint: `context/refcode/FlowMDM/results/humanml/FlowMDM/model000500000.pt` (includes `args.json` with `dataset: "humanml"`).
- HumanML3D normalization arrays (length 263):
  - `HML_Mean_Gen.npy`
  - `HML_Std_Gen.npy`
- These are used by `runners/generate.py -> feats_to_xyz()` when `dataset == 'humanml'`.

## Use the Included HumanML3D Stats
We already vendor the HumanML3D repo with stats:
- `context/refcode/HumanML3D/HumanML3D/Mean.npy`
- `context/refcode/HumanML3D/HumanML3D/Std.npy`

Create FlowMDM‑relative symlinks so its code finds them:
```bash
cd context/refcode/FlowMDM
mkdir -p dataset
# Note: symlink target is relative to the dataset/ folder's parent (FlowMDM/)
ln -sfn ../../HumanML3D/HumanML3D/Mean.npy dataset/HML_Mean_Gen.npy
ln -sfn ../../HumanML3D/HumanML3D/Std.npy  dataset/HML_Std_Gen.npy
ls -la dataset
```
You should see two links pointing to the HumanML3D `Mean.npy` and `Std.npy`.

## Environment Prerequisites
- Install FlowMDM environment and setup extras:
```bash
pixi run flowmdm-install
pixi run flowmdm-setup
```
- Ensure body models are visible to FlowMDM (created once):
```bash
cd context/refcode/FlowMDM
ln -sfn ../../../data/body_models body_models
```

## Run HumanML3D Generation
We provide a Pixi task pinned to the HumanML checkpoint (and HumanML stats):
```bash
pixi run flowmdm-gen-humanml
```
Expected signs of success in the log:
- Loads `./results/humanml/FlowMDM/model000500000.pt`
- Prints `[Inference] BPE switch ... denoising step 60/1000` (typical HumanML setting)
- Saves outputs to `tmp/flowmdm-out/humanml3d/` (`results.npy`, `sample_rep*.mp4`, etc.)

Quick verification:
```bash
pixi run -e dev python - <<'PY'
import numpy as np
D = np.load('tmp/flowmdm-out/humanml3d/results.npy', allow_pickle=True).item()
print(D['motion'].shape, D['motion'].dtype)   # (B, 22, 3, T), float32
print(D['text'], D['lengths'])                # captions + lengths
PY
```

## Troubleshooting
- Stats not found: confirm the symlinks exist inside `context/refcode/FlowMDM/dataset/`.
```bash
pixi run flowmdm-exec -- python - <<'PY'
import os
print('Mean:', os.path.exists('dataset/HML_Mean_Gen.npy'))
print('Std :', os.path.exists('dataset/HML_Std_Gen.npy'))
PY
```
- Wrong checkpoint used (falls back to Babel): `args.json` from the checkpoint overrides CLI options. Ensure `--model_path` points to `results/humanml/FlowMDM/model000500000.pt`.
- Need to generate stats yourself: HumanML3D repo contains notebooks to compute them (see `cal_mean_variance.ipynb`) and README notes; place the resulting `Mean.npy` and `Std.npy` under `HumanML3D/`, then relink as above.

## References
- FlowMDM code that loads stats: `context/refcode/FlowMDM/runners/generate.py`
- Root recovery utilities: `context/refcode/FlowMDM/data_loaders/humanml/scripts/motion_process.py`
- HumanML3D repository (source and docs): https://github.com/EricGuo5513/HumanML3D
- Workspace hint: `context/hints/howto-extract-humanml-root-transform.md` (how to capture root rotation/translation from HumanML3D path)

## Optional: Extract Root Transform (HumanML3D)
To save per‑frame root orientation (quaternion/axis‑angle) and translation during generation, intercept the denormalised features and call `recover_root_rot_pos()`; see the dedicated hint:
- `context/hints/howto-extract-humanml-root-transform.md`
