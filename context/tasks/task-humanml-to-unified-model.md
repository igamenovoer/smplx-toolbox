# Adapting FlowMDM output to Smpl/Smpl-H/Smpl-X models

## Resources
- `FlowMDM` source code: `context/refcode/FlowMDM/`
- how to run `FlowMDM`: see `pyproject.toml`, there are tasks defined to run FlowMDM specifically
- FlowMDM skeleton output format: 
  - `context/refcode/FlowMDM/explain/howto-interpret-flowmdm-output.md`
  - `context/refcode/FlowMDM/explain/about-3d-model-keypoint-topology.md`
  - `context/refcode/FlowMDM/explain/about-smpl-usage-in-flowmdm.md`
- Our unified smpl model:
  - source code: `src/smplx_toolbox/core/unified_model.py`
  - documentation: `docs/unified_model.md`
  - skeleton mapping: `context/hints/smplx-kb/compare-smpl-skeleton.md`

```toml
# pyproject.toml, FlowMDM tasks

# Run commands in FlowMDM dir - use for FlowMDM scripts that need relative paths
flowmdm-exec = { cmd = "cd context/refcode/FlowMDM && pixi run -e latest", description = "Execute arbitrary command in FlowMDM directory with latest environment. Usage: pixi run flowmdm-exec -- <command>" }

# Run commands in workspace with FlowMDM env - use for workspace files needing FlowMDM libraries
flowmdm-exec-local = { cmd = "pixi run --manifest-path context/refcode/FlowMDM/pyproject.toml -e latest", description = "Execute arbitrary command in current directory with FlowMDM environment. Usage: pixi run flowmdm-exec-local -- <command>" }

# Dataset-specific generation helpers (expanded args; run from workspace root)
flowmdm-gen-babel = { cmd = "pixi run flowmdm-exec -- python -m runners.generate-ex --model_path ./results/babel/FlowMDM/model001300000.pt --instructions_file ./tests/simple-walk/simple_walk_instructions.json --num_repetitions 1 --bpe_denoising_step 125 --guidance_param 1.5 --dataset babel --export-smpl --export-smplx --smplx-model-path ./body_models --output_dir ../../../tmp/flowmdm-out/babel", description = "Generate Babel motion (SMPL/SMPL-X export) to tmp/flowmdm-out/babel" }
flowmdm-gen-humanml = { cmd = "pixi run flowmdm-exec -- python -m runners.generate-ex --model_path ./results/babel/FlowMDM/model001300000.pt --instructions_file ./tests/simple-walk/simple_walk_instructions.json --num_repetitions 1 --bpe_denoising_step 125 --guidance_param 1.5 --dataset humanml --output_dir ../../../tmp/flowmdm-out/humanml3d", description = "Generate HumanML3D motion to tmp/flowmdm-out/humanml3d" }
```

## Task 1: Generate motion using FlowMDM

Use the `flowmdm-gen-babel` or `flowmdm-gen-humanml` tasks defined in `pyproject.toml` to generate motion data. The output will include 3D keypoints and, for `babel`, SMPL/SMPL‑X parameters. Redirect outputs to `tmp/flowmdm-out` for further processing.

### Status
- FlowMDM environment installed and set up (SpaCy model + chumpy).
- Created symlink `context/refcode/FlowMDM/body_models -> ../../../data/body_models` for model discovery.
- Ran generation using task: `pixi run flowmdm-gen-babel` (BABEL, simple-walk instructions; SMPL/SMPL‑X export).
- Output written to `tmp/flowmdm-out/babel` including: `results.npy`, `results.txt`, `sample_all.mp4`, `smpl_params.npy`, `smplx_pose.npy`, `smplx_transl.npy`, `smplx_global_orient.npy`, `smplx_global_orient_mat.npy`, `smplx_root_transform.npy`, `smplx_layout.json`.
- Result: motion generated successfully; SMPL‑X path validated; preview videos created.

## Task 2: Understand how FlowMDM uses SMPL/SMPL-H/SMPL-X models

We need to figure out how to convert those outputs to our unified smpl model, the pose part, so that it can animate our unified smpl model, including smpl/smplh/smplx models. We need to transfer the animation in two ways:
- transfer the pose angles (AA format) directly (via `smplx_toolbox.core.NamedPose`), as well as the global orientation and translation, finally create `UnifiedSmplInputs` for each frame.
- transfer via 3D keypoints matching, which is more complex but can handle the extra joints missing in FlowMDM output. This is more robust but slower.
  
### Task 2.1: Generate motion for `babel` and `humanml3d` datasets

You need to first generate motion:
- using the babel dataset, save them into `tmp/flowmdm-out/babel`, then analyze the output files.
- using the humanml3d dataset, save them into `tmp/flowmdm-out/humanml3d`, then analyze the output files.

How to run (from workspace root):

1) Ensure body models symlink exists for FlowMDM

```bash
cd context/refcode/FlowMDM
ln -s ../../../data/body_models body_models  # if not already present
ls -la body_models  # should list smplx/, smplh/, smpl/
cd -
```

2) Generate Babel sample with SMPL/SMPL‑X export into tmp/flowmdm-out/babel

```bash
pixi run flowmdm-gen-babel
```

3) Generate HumanML3D sample into tmp/flowmdm-out/humanml3d

Note: SMPL parameter export is only available for `babel`; for `humanml` you’ll get skeleton/keypoint motion and videos.

```bash
pixi run flowmdm-gen-humanml
```

4) Verify outputs

```bash
ls -la tmp/flowmdm-out/babel
ls -la tmp/flowmdm-out/humanml3d
```

Tip: You can use the predefined tasks `flowmdm-gen-babel` and `flowmdm-gen-humanml` which include expanded args and output directories, or use `flowmdm-exec` manually as shown to customize further.

### Task 2.2: Transfer `babel` SMPL animation to unified smpl model

you have the results in:
- `tmp/flowmdm-out/babel`: generated motion based on `babel` dataset, includes SMPL/SMPL‑X parameters
- `tmp/flowmdm-out/humanml3d`: generated motion based on `humanml3d` dataset, includes skeleton/keypoint motion

for babel output, we shall construct a list of `UnifiedSmplInputs` objects, one for each frame, each such object only contains:
- `named_pose`: intrinsic pose only (no pelvis), with appropriate `model_type` (smpl/smplh/smplx) and `batch_size=1`
- `global_orient`: pelvis AA
- `transl`: global translation
- for other fields, remain `None`

the converted `UnifiedSmplInputs` should be stored as `unify_smpl_animation.pkl` in the same dir (`tmp/flowmdm-out/babel`), so that it can be loaded later for animation.

implementation should be a CLI tool in `scripts/cvt_flowmdm_babel_to_smpl_animation.py`, which takes:
- `--input_dir`: input dir containing FlowMDM babel output (default: `tmp/flowmdm-out/babel`)
- `--output_path`: output path to save the pickled list of `UnifiedSmplInputs` (default: same dir as input, filename `unified_smpl_animation.pkl`)

How to run (from workspace root):

```bash
pixi run -e dev python scripts/cvt_flowmdm_babel_to_smpl_animation.py \
  --input_dir tmp/flowmdm-out/babel \
  --output_path tmp/flowmdm-out/babel/unified_smpl_animation.pkl
```

Notes:
- Uses SMPL-H as the default `model_type` and fills body joints; missing hands/face remain zeros.
- Falls back to `smplx_pose.npy` + `smplx_global_orient.npy` + `smplx_transl.npy` if `smpl_params.npy` is absent.

Status:
- [x] Implemented converter: `scripts/cvt_flowmdm_babel_to_smpl_animation.py`.
- [x] Generated output: `tmp/flowmdm-out/babel/unified_smpl_animation.pkl`.

### Task 2.3: Visualize the converted `babel` animation

we need to visualize the converted `babel` animation (`tmp/flowmdm-out/babel/unified_smpl_animation.pkl`), using `pyvista` for rendering.

**Deliverable**: a CLI tool `scripts/show-animation-unified-model.py` that:
- `--anim-file <filepath>`: path to the pickled list of `UnifiedSmplInputs` (required)
- `--model-type <smpl/smplh/smplx>`: model type to visualize (default: `smplx`)
- `--body-model-dir <dir>`: path to body models (default: `data/body_models`)
- `--backend <basic/qt/browser>`: rendering backend (default: `basic`)
  - for `basic`, use on-screen PyVista rendering with a turntable-style camera
  - for `qt`, use `pyvistaqt` (requires `pyqt5` and XCB on Linux)
  - for `browser`, use Trame/VTK to render in the browser; optionally provide `--port`, otherwise a free port is auto-picked
- Interactive control:
  - `Left/Right`: step backward/forward one frame
  - `r`: reset view

Status:
- [x] Implemented CLI with `basic`, `qt`, and `browser` backends
- [x] Added back-compat alias `--body-models-path` for existing tasks
- [x] Added optional `--port` flag for the browser backend (auto-pick when omitted)
- [x] Added global axis / 10×10 m ground grid and verified browser reachability
- [x] Updated Pixi task `flowmdm-show-babel-anim` to use `--backend browser`
- [x] **Task complete**: viewer matches requirements and confirmed on WSL

How to run (from workspace root):

```bash
# Launch browser viewer on the converted Babel animation
pixi run flowmdm-show-babel-anim

# Or manually with explicit flags
pixi run -e dev python scripts/show-animation-unified-model.py \
  --anim-file tmp/flowmdm-out/babel/unified_smpl_animation.pkl \
  --body-model-dir data/body_models \
  --model-type smplx \
  --backend browser

# Override port (e.g., when tunnelling through SSH)
pixi run -e dev python scripts/show-animation-unified-model.py \
  --anim-file tmp/flowmdm-out/babel/unified_smpl_animation.pkl \
  --backend browser --port 9000
```

### Task 2.4: Visualize the converted `humanml3d` animation

- see `context/refcode/FlowMDM/explain/howto-interpret-flowmdm-output.md`, the humanml3d output does not include smpl/smplx parameters, only 3D keypoints.
- flowmdm motion generation script: `context/refcode/FlowMDM/runners/generate-ex.py`

as such, we need to:
- find out whether global translation and orientation have been computed separately in the original script, and if so, how to extract them.

Status:
- [x] Confirmed source of global yaw+translation in HumanML3D path: `recover_root_rot_pos()` inside `data_loaders/humanml/scripts/motion_process.py`; added extraction guide `context/hints/howto-extract-humanml-root-transform.md`.
- [x] Added setup guide `context/hints/howto-setup-humanml3d.md` and wired required HumanML stats:
  - Created symlinks expected by FlowMDM: `context/refcode/FlowMDM/dataset/HML_Mean_Gen.npy` and `.../HML_Std_Gen.npy` → point to `context/refcode/HumanML3D/HumanML3D/{Mean,Std}.npy`.
- [x] Updated Pixi tasks to actually use HumanML checkpoint and kept a fallback:
  - `flowmdm-gen-humanml` → uses `./results/humanml/FlowMDM/model000500000.pt` and HumanML stats.
  - `flowmdm-gen-humanml-via-babel` → legacy behavior using Babel checkpoint, writes to the same output folder.
- [x] Verified HumanML3D generation runs and writes to `tmp/flowmdm-out/humanml3d` (`results.npy` shape `[B,22,3,T]`).
- [ ] Converter from HumanML3D joints → `UnifiedSmplInputs` (pose fitting/IK) pending; plan is to fit SMPL‑X `(global_orient, body_pose, transl)` against T2M joints (see hints for root-only reconstruction as interim).

How to run (HumanML3D):
```bash
# Ensure stats are linked once
cd context/refcode/FlowMDM
mkdir -p dataset
ln -sfn ../../HumanML3D/HumanML3D/Mean.npy dataset/HML_Mean_Gen.npy
ln -sfn ../../HumanML3D/HumanML3D/Std.npy  dataset/HML_Std_Gen.npy

# Generate
pixi run flowmdm-gen-humanml

# Quick inspect
pixi run -e dev python - <<'PY'
import numpy as np
D = np.load('tmp/flowmdm-out/humanml3d/results.npy', allow_pickle=True).item()
print(D['motion'].shape, D['motion'].dtype)
print(D['text'], D['lengths'])
PY
```

Notes:
- The Babel converter `scripts/cvt_flowmdm_babel_to_smpl_animation.py` is not applicable to HumanML3D (no SMPL params). Use fitting to recover SMPL‑X parameters from joints, or a root‑only viewer using recovered quaternion + translation.
