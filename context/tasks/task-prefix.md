# Information about current status of the project

## Python Environment

- we are using `pixi`, to run any pythong script, use `pixi run -e dev <your-script.py>`
  
- If you want to run inline python code, use single-quotes in outer layer and double-quotes in inner layer, like this:

```powershell
pixi run -e dev python -c 'print("Hello World")'
```

- DO NOT try to install python packages, ask the admin for help.

## Resource Locations

you can find the resources in the following locations.

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