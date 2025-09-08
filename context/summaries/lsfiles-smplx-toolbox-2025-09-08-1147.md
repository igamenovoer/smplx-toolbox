# SMPL-X Toolbox File Structure (Depth 2)
*Generated: 2025-09-08 11:47*

## Directory Tree Overview

```
smplx-toolbox/
├── .editorconfig              # Editor formatting configuration
├── .env                       # Environment variables
├── .gitattributes            # Git line ending configuration  
├── .gitignore                # Git ignore patterns
├── .gitmodules               # Git submodule definitions
├── CLAUDE.md                 # AI assistant guidelines for Claude
├── LICENSE                   # MIT license file
├── README.md                 # Main project documentation
├── ROADMAP.md                # Development roadmap
├── mkdocs.yml                # MkDocs documentation configuration
├── pixi.lock                 # Pixi dependency lock file
├── pyproject.toml            # Python package configuration
│
├── .github/                  # GitHub-specific files
│   └── workflows/            # CI/CD workflow definitions
│
├── .magic-context/           # Magic Context submodule for AI workflows
│   ├── LICENSE               # License for Magic Context
│   ├── README.md             # Magic Context documentation
│   ├── blender-plugin/       # Blender plugin templates
│   ├── general/              # General templates
│   ├── instructions/         # AI instruction templates
│   ├── mcp-dev/              # MCP development templates
│   ├── roles/                # AI role definitions
│   └── scripts/              # Utility scripts
│
├── .mypy_cache/              # MyPy type checking cache
│   ├── .gitignore            # Cache ignore patterns
│   ├── 3.11/                 # Python 3.11 type cache
│   ├── CACHEDIR.TAG          # Cache directory marker
│   └── missing_stubs/        # Missing type stubs
│
├── .pixi/                    # Pixi environment management
│   ├── .condapackageignore   # Conda package ignore patterns
│   ├── .gitignore            # Pixi ignore patterns
│   └── envs/                 # Virtual environments
│
├── .promptx/                 # PromptX AI configuration
│   ├── .xml-upgrade-backup-done  # XML upgrade marker
│   ├── backup/               # Configuration backups
│   ├── memory/               # AI memory storage
│   ├── pouch.json            # PromptX configuration
│   └── resource/             # AI resources
│
├── .ruff_cache/              # Ruff linter cache
│   ├── .gitignore            # Cache ignore patterns
│   ├── 0.12.11/              # Ruff version cache
│   └── CACHEDIR.TAG          # Cache directory marker
│
├── context/                  # AI assistant workspace
│   ├── README.md             # Context directory guide
│   ├── design/               # Technical specifications
│   ├── hints/                # Development hints
│   ├── instructions/         # Work instructions
│   ├── logs/                 # Development logs
│   ├── plans/                # Implementation plans
│   ├── refcode/              # Reference code (3 major SMPL-X implementations)
│   │   ├── human_body_prior/ # VPoser - Variational pose prior
│   │   ├── smplify-x/        # 2D image to SMPL-X fitting
│   │   └── smplx/            # Official SMPL-X PyTorch implementation
│   ├── roles/                # Role configurations
│   ├── summaries/            # Analysis summaries
│   ├── tasks/                # Task definitions
│   └── tools/                # Development tools
│
├── data/                     # Data storage
│   ├── README.md             # Data directory documentation
│   └── body_models/          # SMPL-X model files
│
├── docs/                     # Documentation source
│   ├── configuration-summary.md  # Config documentation
│   ├── index.md              # Documentation homepage
│   ├── installation.md       # Installation guide
│   ├── line-endings.md       # Line ending guide
│   ├── pypi-pytorch-strategy.md  # PyTorch packaging strategy
│   └── pytorch-installation.md    # PyTorch installation guide
│
├── scripts/                  # Command-line tools
│   ├── auto-install.py       # Auto-installation script
│   └── smplx-toolbox         # CLI executable
│
├── src/                      # Source code
│   └── smplx_toolbox/        # Main package
│       ├── __init__.py       # Package initialization
│       ├── core/             # Core functionality
│       ├── optimization/     # Optimization algorithms
│       ├── utils/            # Utility functions
│       └── visualization/    # Visualization tools
│
├── tests/                    # Test suite
│   └── README.md             # Testing documentation
│
└── tmp/                      # Temporary working files
    ├── check_api.py          # API checking script
    ├── check_deps.py         # Dependency checker
    ├── check_torch.py        # PyTorch verification
    ├── mesh_exports/         # Exported mesh files
    ├── revised_test/         # Revised test outputs
    ├── ruff_benefits.md      # Ruff linter benefits
    ├── smplh_export/         # SMPL-H export files
    ├── SMPLH_MODEL_SETUP.md  # SMPL-H setup guide
    ├── SMPLH_REAL_TEST_SUMMARY.md  # Test results
    ├── smplh_test/           # SMPL-H test files
    ├── smplx_model_revision_summary.md  # Model revision notes
    ├── smplx_tests/          # SMPL-X test files
    ├── test_real_smplh.py    # Real SMPL-H tests
    ├── test_revised_model.py # Revised model tests
    ├── test_smplh_model.py   # SMPL-H model tests
    ├── test_smplh_wrapper.py # SMPL-H wrapper tests
    ├── test_to_mesh.py       # Mesh conversion tests
    └── visualize_mesh.py     # Mesh visualization script

```

## File Descriptions

### Root Configuration Files

#### `.editorconfig`
Editor configuration ensuring consistent code formatting across different IDEs and editors.

#### `.env`
Environment variables for local development configuration.

#### `.gitattributes`
Git configuration for handling line endings consistently across platforms (CRLF/LF normalization).

#### `.gitignore`
Specifies files and directories Git should ignore (caches, build artifacts, environments).

#### `.gitmodules`
Defines the Magic Context submodule reference for AI development workflows.

#### `CLAUDE.md`
Comprehensive guidelines for Claude AI assistant when working with this codebase. Includes project structure, development commands, coding standards, and the unique member variable naming conventions (m_ prefix).

#### `LICENSE`
MIT license defining the terms for using and distributing this software.

#### `README.md`
Main project documentation with installation instructions, features, and usage examples.

#### `ROADMAP.md`
Strategic development roadmap outlining planned features and milestones.

#### `mkdocs.yml`
Configuration for MkDocs static site generator used for project documentation.

#### `pixi.lock`
Lock file ensuring reproducible Pixi environment installations.

#### `pyproject.toml`
Python package configuration defining:
- Package metadata (name, version, description)
- Dependencies (torch, numpy, trimesh, etc.)
- Development tools (pytest, ruff, mypy)
- Build system configuration

### Hidden Directories

#### `.github/workflows/`
GitHub Actions CI/CD workflows for automated testing and deployment.

#### `.magic-context/`
Git submodule containing AI development templates and best practices for structured AI-assisted development.

#### `.mypy_cache/`
Type checking cache for MyPy static type analyzer, improving performance of type checking operations.

#### `.pixi/`
Pixi package manager's virtual environment and configuration storage.

#### `.promptx/`
PromptX AI assistant configuration including:
- Role definitions for specialized AI personas
- Memory storage for context retention
- Resource registry for project-specific knowledge

#### `.ruff_cache/`
Cache for Ruff Python linter, speeding up code quality checks.

### Main Directories

#### `context/`
AI assistant workspace following the Context Directory Guide pattern:
- **design/**: Technical architecture and design documents
- **hints/**: Knowledge base and troubleshooting guides
- **instructions/**: Task-specific work instructions
- **logs/**: Development session records
- **plans/**: Implementation strategies and roadmaps
- **refcode/**: Reference implementations and examples (see detailed breakdown below)
- **roles/**: AI role configurations (animation-dev, etc.)
- **summaries/**: Analysis and consolidated knowledge
- **tasks/**: Current and planned work items
- **tools/**: Development utilities and scripts

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

#### `docs/`
Documentation source files:
- **configuration-summary.md**: Project configuration overview
- **index.md**: Documentation homepage
- **installation.md**: Step-by-step installation guide
- **line-endings.md**: Cross-platform line ending handling
- **pypi-pytorch-strategy.md**: Strategy for PyTorch packaging
- **pytorch-installation.md**: PyTorch-specific setup instructions

#### `scripts/`
Command-line interface tools:
- **auto-install.py**: Automated dependency installation script
- **smplx-toolbox**: Main CLI executable for toolbox operations

#### `src/smplx_toolbox/`
Main package source code:

##### `__init__.py`
Package initialization exposing version info and main imports.

##### `core/`
Core functionality module:
- **smplx_model.py**: 
  - `SMPLXModel` class: Main SMPL-X model wrapper
  - Methods: `from_smplx()`, `to_mesh()`, `get_joint_positions()`
  - Properties: `faces`, `num_joints` (55), `num_vertices` (10475), `joint_names`
  
- **smplh_model.py**:
  - `SMPLHModel` class: SMPL+H model implementation
  - Similar interface to SMPLXModel but for hand-focused models

##### `optimization/`
Parameter optimization algorithms (placeholder for future implementation).

##### `utils/`
Utility functions for data processing and format conversion.

##### `visualization/`
3D visualization and rendering tools (placeholder for future implementation).

#### `tests/`
Test suite with pytest-based unit and integration tests.

#### `tmp/`
Temporary working directory containing:
- **Test scripts**: Various model testing implementations
- **check_*.py**: Verification scripts for API, dependencies, PyTorch
- **Export directories**: mesh_exports/, smplh_export/, test outputs
- **Documentation**: SMPL-H setup guides and test summaries
- **visualize_mesh.py**: Mesh visualization utility

## Key Technical Details

### Python Package Structure
- Uses modern Python packaging with `pyproject.toml`
- Follows PEP 517/518 standards
- Supports editable installations for development

### Development Environment
- **Pixi**: Cross-platform package management
- **Ruff**: Fast Python linter replacing Flake8/Black/isort
- **MyPy**: Static type checking for type safety
- **Pytest**: Testing framework with coverage reports

### Coding Standards
- Member variables prefixed with `m_`
- Read-only properties via `@property` decorators
- Explicit setter methods for state changes
- Factory methods for object creation (`from_*()`)
- NumPy-style docstrings

### Model Support
- SMPL-X: Full body with hands and face (55 joints)
- SMPL-H: Body with detailed hands
- PyTorch-based for GPU acceleration
- Trimesh integration for mesh operations

### AI Integration
- PromptX for specialized AI roles
- Context Directory pattern for knowledge management
- Magic Context templates for structured development
- Animation-dev role for 3D/Blender expertise

## Development Commands

Key Pixi commands (from project root):
```bash
pixi install          # Install dependencies
pixi run test         # Run tests
pixi run lint         # Run linting
pixi run format       # Format code
pixi run type-check   # Type checking
pixi run docs-serve   # Serve documentation
```

## File Size Considerations
- Model files (.pkl) can be 100+ MB
- Cache directories can grow large
- tmp/ directory should be cleaned periodically
- Use .gitignore to exclude large/generated files