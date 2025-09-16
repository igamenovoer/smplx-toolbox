# How to run FlowMDM and redirect its output (with SMPL/SMPL‑X models)

This guide shows how to:
- Install and set up the FlowMDM environment via Pixi.
- Generate motion with SMPL/SMPL‑X exports.
- Redirect outputs to a workspace path like `tmp/flowmdm-out`.
- Point FlowMDM at your body model files so it can validate and export meshes.

## Prerequisites
- Ensure you have the SMPL/SMPL‑H/SMPL‑X body model assets available locally under `data/body_models/`:
  - Expected directory structure: `data/body_models/smplx/` containing model files (e.g., `SMPLX_NEUTRAL.npz` or `.pkl`).
  - Download SMPL‑X models from the official site: https://smpl-x.is.tue.mpg.de/

## 0) Create a `body_models` symlink inside FlowMDM
FlowMDM defaults to `./body_models/...` (see `utils/config.py`). Create a symlink so FlowMDM can find the repo’s shared assets in `data/body_models`:

```bash
# From the workspace root
cd context/refcode/FlowMDM
ln -s ../../../data/body_models body_models

# Verify (should list smplx, smplh, smpl, etc.)
ls -la body_models
```

Notes:
- On Windows (PowerShell), use: `New-Item -ItemType SymbolicLink -Path body_models -Target ..\..\..\data\body_models`
- On Windows (cmd.exe as Administrator), use: `mklink /D body_models ..\..\..\data\body_models`

After this, you can pass `--smplx-model-path ./body_models` to FlowMDM.

## 1) Install FlowMDM environment and setup
Run the FlowMDM tasks from the workspace root:

```bash
# Install FlowMDM's latest environment
pixi run flowmdm-install

# Download SpaCy model and install legacy chumpy
pixi run flowmdm-setup
```

Useful checks:
```bash
# Optional: verify PyTorch CUDA in the FlowMDM env
pixi run flowmdm-test-cuda

# Optional: show available tasks in FlowMDM
pixi run flowmdm-list-tasks
```

## 2) Generate motion with SMPL/SMPL‑X export
The extended generator supports exporting SMPL parameters and building SMPL‑X‑aligned poses. Two ways to run it:

Option A — use the FlowMDM env from within its directory (preferred; uses symlink):
```bash
# This runs from context/refcode/FlowMDM using its latest env
pixi run flowmdm-exec -- \
  python -m runners.generate-ex \
    --model_path ./results/babel/FlowMDM/model001300000.pt \
    --instructions_file ./tests/simple-walk/simple_walk_instructions.json \
    --num_repetitions 1 \
    --bpe_denoising_step 125 \
    --guidance_param 1.5 \
    --dataset babel \
    --export-smpl \
    --export-smplx \
    --smplx-model-path ./body_models \
    --output_dir ../../../tmp/flowmdm-out
```

Option B — dataset-specific tasks (preconfigured):
```bash
# Babel (exports SMPL/SMPL-X) → tmp/flowmdm-out/babel
pixi run flowmdm-gen-babel

# HumanML3D (skeleton/videos) → tmp/flowmdm-out/humanml3d
pixi run flowmdm-gen-humanml
```

## 3) How FlowMDM locates body models
- The flag `--smplx-model-path` must point to the PARENT directory that contains the `smplx/` folder (not the folder itself).
  - With the symlink above, simply pass `--smplx-model-path ./body_models`.
  - Without the symlink, in this repo the parent is `data/body_models` (from FlowMDM dir: `../../../data/body_models`).

Examples:
```bash
# Using the symlink from FlowMDM directory
--smplx-model-path ./body_models

# If you stay at workspace root and need FlowMDM libs for your own script
pixi run flowmdm-exec-local -- python your_script.py --smplx-model-path ./data/body_models
```

If the path is wrong, FlowMDM will error with a message similar to:
```
[error] SMPLX model directory not found at <path>/smplx
This should be the PARENT directory containing 'smplx' folder.
Example: use '../../data' instead of '../../data/smplx'
```

## 4) Outputs
With `--output_dir ../../../tmp/flowmdm-out`, you should see (examples):
```text
tmp/flowmdm-out/
  results.npy, results.txt
  smpl_params.npy
  smplx_pose.npy, smplx_transl.npy
  smplx_global_orient.npy, smplx_global_orient_mat.npy
  smplx_root_transform.npy, smplx_layout.json
  sample_all.mp4, sample_rep00.mp4
```

## 5) Optional: visualize the SMPL‑X mesh
```bash
pixi run flowmdm-exec -- \
  pixi run -e latest python visualization/show-animation-smplx.py \
    ../../../tmp/flowmdm-out \
    --smplx-model-path ../../../data/body_models \
    --autoplay
```

## References
- FlowMDM (official repository): https://github.com/yufu-wang/FlowMDM
- SMPL‑X model downloads: https://smpl-x.is.tue.mpg.de/
- In‑repo explainers:
  - `context/refcode/FlowMDM/explain/howto-interpret-flowmdm-output.md`
  - `context/refcode/FlowMDM/explain/howto-export-smpl-params-and-convert-to-smplx.md`
