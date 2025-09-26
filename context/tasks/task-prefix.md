# Information about current status of the project

## Python Environment

- IMPORTANT: we are using `pixi`, to run any pythong script, use `pixi run -e dev <your-script.py>`
  
- If you want to run inline python code, use single-quotes in outer layer and double-quotes in inner layer, like this:

```powershell
pixi run -e dev python -c 'print("Hello World")'
```

- DO NOT try to install python packages, ask the admin for help.

## Resource Locations

you can find the resources in the following locations.

#### SMPL series model files
- SMPL: `data/body_models/smpl/SMPL_NEUTRAL.pkl`
- SMPL-H: `data/body_models/smplh/SMPLH_MALE.pkl`
- SMPL-H-mano: `data/body_models/mano_v1_2/models/SMPLH_male.pkl`
- SMPL-H (NPZ variant): `data/body_models/smplh/male/model.npz`
- SMPL-X (PKL): `data/body_models/smplx/SMPLX_MALE.pkl`
- SMPL-X (NPZ): `data/body_models/smplx/SMPLX_MALE.npz`

#### SMPL model relationships
- overview: `context/hints/smplx-kb/compare-smpl-models.md`
- skeleton: `context/hints/smplx-kb/compare-smpl-skeleton.md`
- shapedirs: `context/hints/smplx-kb/compare-smpl-shape.md`
- conversion: `context/hints/smplx-kb/about-smplx-transfer-model.md`

##### `context/refcode/` (Detailed Breakdown)
Reference code repository containing three major SMPL-X related implementations:

###### **1. human_body_prior/**
VPoser - Variational Human Pose Prior for SMPL body models
- **Purpose**: Provides learned priors for human pose, enabling more realistic pose generation and optimization
- **Key Components**:
  - `src/human_body_prior/body_model/`: SMPL body model implementations
    - `body_model.py`: Core body model wrapper
    - `lbs.py`: Linear Blend Skinning implementation
    - `parts_segm/`: Body part segmentation data
  - `src/human_body_prior/models/`: VPoser and IK models
    - `vposer_model.py`: Variational pose prior neural network
    - `ik_engine.py`: Inverse kinematics solver
  - `src/human_body_prior/tools/`: Utility functions
    - `rotation_tools.py`: Rotation conversions and utilities
    - `bodypart2vertexid.py`: Body part to vertex mapping
  - `src/human_body_prior/train/`: Training scripts for VPoser
  - `tutorials/`: Jupyter notebooks and examples
    - `vposer.ipynb`: VPoser usage tutorial
    - `ik_example_*.py`: IK examples for joints and mocap

###### **2. smplify-x/**
Expressive Body Capture: 3D Hands, Face, and Body from a Single Image
- **Purpose**: Fits SMPL-X models to 2D images using optimization
- **Key Components**:
  - `smplifyx/`: Main fitting implementation
    - `main.py`: Entry point for fitting pipeline
    - `fitting.py`: Core optimization routines
    - `camera.py`: Camera model and projection
    - `prior.py`: Body pose and shape priors
  - `smplifyx/optimizers/`: Custom optimizers
    - `lbfgs_ls.py`: Line-search LBFGS optimizer
  - `cfg_files/`: Configuration files
    - `fit_smpl.yaml`: SMPL fitting config
    - `fit_smplh.yaml`: SMPL-H fitting config
    - `fit_smplx.yaml`: SMPL-X fitting config

###### **3. smplx/** (Official Implementation)
SMPL-X: A unified model for expressive body, face, and hands
- **Purpose**: The official PyTorch implementation of SMPL, SMPL-H, SMPL-X, MANO, and FLAME models
- **Key Components**:
  - `smplx/`: Core model implementations
    - `body_models.py`: SMPL, SMPL-H, SMPL-X, MANO, FLAME classes
    - `lbs.py`: Linear Blend Skinning with corrective blend shapes
    - `joint_names.py`: Joint naming conventions across models
    - `vertex_ids.py`: Vertex indices for landmarks
    - `vertex_joint_selector.py`: Joint and vertex selection utilities
  - `examples/`: Usage examples
    - `demo.py`: Basic model usage demo
    - `vis_*_vertices.py`: Visualization scripts for different models
  - `transfer_model/`: Model conversion between formats
    - `transfer_model.py`: Main transfer script
    - `config/`: Transfer configuration
    - `data/datasets/mesh.py`: Mesh dataset handling
    - `losses/`: Loss functions for model fitting
    - `optimizers/`: Optimization routines
    - `utils/`: Transfer utilities
      - `def_transfer.py`: Deformation transfer
      - `mesh_utils.py`: Mesh processing
      - `pose_utils.py`: Pose manipulation
  - `transfer_data/`: Pre-computed transfer matrices
    - `smpl2smplx_*.pkl`: SMPL to SMPL-X transfer
    - `smplh2smplx_*.pkl`: SMPL-H to SMPL-X transfer
    - Various deformation transfer setups
  - `config_files/`: Model conversion configs
  - `tools/`: Utility scripts
    - `merge_smplh_mano.py`: Merge SMPL-H with MANO hands
    - `clean_ch.py`: Clean Chumpy dependencies

#### `data/`
Data storage directory:
- **body_models/**: SMPL-X model files (.pkl format)
- Model downloads and preprocessing artifacts
- **vposer/**: VPoser model files `vposer-v2.ckpt` for v2 vposer model checkpoint.

#### More about FlowMDM and HumanML3D
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