# SMPL-X Knowledge Base

## Overview

SMPL-X (SMPL eXpressive) is a unified 3D parametric model of the human body, face, and hands. It provides a differentiable layer that outputs 3D meshes of the human body controlled by pose, shape, and expression parameters.

## Model Family

The SMPL model family consists of:
- **SMPL**: Base body model (10 shape params, 23 joints)
- **SMPL+H (SMPLH)**: Body + detailed hands
- **SMPL-X**: Body + hands + face + expressions
- **MANO**: Standalone hand model  
- **FLAME**: Standalone head/face model

## Installation

```bash
# From PyPI
pip install smplx[all]

# From source
git clone https://github.com/vchoutas/smplx
python setup.py install
```

## Core Classes and Interfaces

### SMPL Class
Base body model with 23 joints.

```python
class SMPL(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300
    
    def __init__(
        self,
        model_path: str,                    # Path to model files
        kid_template_path: str = '',        # Optional kid template
        num_betas: int = 10,                # Shape coefficients  
        create_global_orient: bool = True,  # Create orientation param
        create_body_pose: bool = True,      # Create pose params
        create_transl: bool = True,         # Create translation param
        gender: str = 'neutral',            # Gender: neutral/male/female
        age: str = 'adult',                 # Age: adult/kid/baby
        batch_size: int = 1,
        dtype=torch.float32,
        **kwargs
    )
    
    def forward(
        self,
        betas: Optional[Tensor] = None,     # Shape params (B, 10)
        body_pose: Optional[Tensor] = None, # Body pose (B, 69) 
        global_orient: Optional[Tensor] = None, # Root orient (B, 3)
        transl: Optional[Tensor] = None,    # Translation (B, 3)
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,              # Convert axis-angle to rot
        **kwargs
    ) -> SMPLOutput
```

### SMPLX Class
Full expressive model with body, hands, and face.

```python
class SMPLX(SMPLH):
    NUM_JOINTS = 55
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    
    def __init__(
        self,
        model_path: str,
        num_expression_coeffs: int = 10,    # Expression components
        create_expression: bool = True,      
        create_jaw_pose: bool = True,       # Jaw articulation
        create_leye_pose: bool = True,      # Left eye pose
        create_reye_pose: bool = True,      # Right eye pose  
        use_face_contour: bool = False,     # Facial contour keypoints
        use_pca: bool = True,                # PCA for hand pose
        num_pca_comps: int = 6,              # PCA components
        flat_hand_mean: bool = False,       # Flat vs natural hand
        **kwargs
    )
    
    def forward(
        self,
        betas: Optional[Tensor] = None,     # Shape (B, 10)
        global_orient: Optional[Tensor] = None, # Root (B, 3)
        body_pose: Optional[Tensor] = None, # Body (B, 63)
        left_hand_pose: Optional[Tensor] = None,  # Left hand
        right_hand_pose: Optional[Tensor] = None, # Right hand
        expression: Optional[Tensor] = None, # Face expression (B, 10)
        jaw_pose: Optional[Tensor] = None,   # Jaw (B, 3)
        leye_pose: Optional[Tensor] = None,  # Left eye (B, 3)
        reye_pose: Optional[Tensor] = None,  # Right eye (B, 3)
        transl: Optional[Tensor] = None,     # Translation (B, 3)
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        return_shaped: bool = True,          # Return v_shaped
        **kwargs
    ) -> SMPLXOutput
```

### Output Data Structures

```python
# SMPL Output
SMPLOutput = namedtuple(
    'SMPLOutput',
    ['vertices',      # (B, 6890, 3) mesh vertices
     'joints',        # (B, J, 3) joint positions  
     'betas',         # (B, 10) shape parameters
     'body_pose',     # (B, 69) body pose
     'global_orient', # (B, 3) global orientation
     'full_pose',     # Optional: full pose vector
     'v_shaped']      # Optional: shaped vertices
)

# SMPL-X Output  
SMPLXOutput = namedtuple(
    'SMPLXOutput',
    ['vertices',      # (B, 10475, 3) mesh vertices
     'joints',        # (B, 127, 3) joints + landmarks
     'betas',         # (B, 10) shape parameters
     'expression',    # (B, 10) expression params
     'global_orient', # (B, 3) global orientation
     'body_pose',     # (B, 63) body pose
     'left_hand_pose', # Left hand parameters
     'right_hand_pose', # Right hand parameters  
     'jaw_pose',      # (B, 3) jaw pose
     'v_shaped',      # Shaped template vertices
     'full_pose']     # Optional: full pose vector
)
```

## Model Creation

### Using the Factory Function

```python
import smplx

# Create SMPL-X model
model = smplx.create(
    model_path='path/to/models',
    model_type='smplx',        # smpl, smplh, smplx, mano, flame
    gender='neutral',           # neutral, male, female
    use_face_contour=True,
    num_betas=10,
    num_expression_coeffs=10,
    use_pca=False,              # Full hand joints vs PCA
    batch_size=1
)

# Forward pass with parameters
output = model(
    betas=betas_tensor,         # Shape: (batch, 10)
    body_pose=body_pose_tensor, # Shape: (batch, 21, 3)
    global_orient=orient_tensor,
    expression=expr_tensor,
    return_verts=True
)

vertices = output.vertices     # (batch, 10475, 3)
joints = output.joints         # (batch, 127, 3)
```

### Using Layer Classes (No Parameters)

```python
# For integration into neural networks
layer = smplx.SMPLXLayer(
    model_path='path/to/models',
    gender='neutral'
)

# All parameters must be provided
output = layer(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient,
    expression=expression
)
```

## Model Parameters

### Shape Parameters (β)
- **SMPL/SMPL-X**: 10 PCA components controlling body shape
- Range: typically [-5, 5] for realistic shapes
- Dimension: (batch_size, 10)

### Pose Parameters (θ)  
- **Axis-angle format**: 3 values per joint (rotation axis * angle)
- **SMPL**: 23 joints × 3 = 69 values
- **SMPL-X**: 55 joints × 3 = 165 values 
  - Body: 21 joints (63 values)
  - Hands: 15 joints × 2 = 30 joints (90 values)
  - Face: jaw + eyes = 3 joints (9 values)

### Expression Parameters (ψ)
- **SMPL-X only**: 10 PCA components for facial expressions
- Controls facial deformations beyond jaw movement
- Dimension: (batch_size, 10)

### Hand Pose Representation
```python
# PCA mode (default, more stable)
model = smplx.create(..., use_pca=True, num_pca_comps=6)
left_hand_pose = torch.randn(batch, 6)  # 6 PCA components

# Full joint mode (more control)
model = smplx.create(..., use_pca=False)  
left_hand_pose = torch.randn(batch, 15, 3)  # 15 joints
```

## Key Features

### Linear Blend Skinning (LBS)
The core deformation function:
```python
def lbs(betas, pose, v_template, shapedirs, posedirs, 
        J_regressor, parents, lbs_weights):
    # 1. Apply shape blend shapes
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    
    # 2. Compute joint locations
    J = vertices2joints(J_regressor, v_shaped)
    
    # 3. Apply pose blend shapes  
    v_posed = v_shaped + posedirs @ pose_features
    
    # 4. Skinning deformation
    T = compute_transforms(pose, J, parents)
    v_final = skinning(v_posed, T, lbs_weights)
    
    return v_final, J
```

### Joint Hierarchy
SMPL-X follows a kinematic tree structure:
```
pelvis (root)
├── left_hip → left_knee → left_ankle → left_foot
├── right_hip → right_knee → right_ankle → right_foot
└── spine1 → spine2 → spine3
    ├── neck → head → jaw/eyes
    ├── left_collar → left_shoulder → ... → left_fingers
    └── right_collar → right_shoulder → ... → right_fingers
```

### Landmarks and Keypoints
SMPL-X provides various anatomical landmarks:
- Body joints: 55 skeletal joints
- Hand landmarks: 21 per hand (when using MANO)
- Face landmarks: 51 facial keypoints
- Optional face contour: 17 dynamic contour points

## Model Files and Formats

### Directory Structure
```
models/
├── smpl/
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smplx/
│   ├── SMPLX_FEMALE.npz  # Recommended format
│   ├── SMPLX_MALE.npz
│   └── SMPLX_NEUTRAL.npz
└── mano/
    ├── MANO_LEFT.pkl
    └── MANO_RIGHT.pkl
```

### Model Data Contents
- `v_template`: Template mesh vertices (rest pose)
- `shapedirs`: Shape blend shapes
- `posedirs`: Pose blend shapes  
- `J_regressor`: Joint regressor matrix
- `lbs_weights`: Skinning weights
- `faces`: Triangle face connectivity

## Advanced Usage

### Coordinate System Transformation
SMPL-X uses Y-up coordinate system by default. For Blender/Unity (Z-up):

```python
def transform_to_z_up(vertices):
    """Convert from Y-up to Z-up coordinate system"""
    x, y, z = vertices[..., 0], vertices[..., 1], vertices[..., 2]
    return torch.stack([x, -z, y], dim=-1)
```

### Batch Processing
```python
# Process multiple poses efficiently
batch_size = 32
betas = torch.randn(batch_size, 10)
body_pose = torch.randn(batch_size, 63)

output = model(betas=betas, body_pose=body_pose)
# output.vertices shape: (32, 10475, 3)
```

### Model Transfer Between Formats
```bash
# SMPL to SMPL-X
python -m transfer_model --exp-cfg config_files/smpl2smplx.yaml

# SMPL-X to SMPL (loses hand/face detail)  
python -m transfer_model --exp-cfg config_files/smplx2smpl.yaml
```

## Common Pitfalls and Solutions

### Issue: Incorrect Pose Dimensions
```python
# Wrong: flat pose vector
pose = torch.randn(batch, 69)  

# Correct: reshape to (joints, 3)
pose = torch.randn(batch, 23, 3)
# Or let the model handle it
pose = torch.randn(batch, 69)
model(body_pose=pose, pose2rot=True)  # pose2rot handles conversion
```

### Issue: Missing Dependencies
```python
# RuntimeError: No module named 'smplx'
# Solution: Install with dependencies
pip install smplx[all]
```

### Issue: Model Files Not Found
```python
# Ensure correct path and file naming
model_path = '/path/to/models'  # Parent directory
# Files should be named: SMPLX_NEUTRAL.npz, etc.
```

## Performance Optimization

### Caching Neutral Pose
```python
# Cache the neutral/T-pose for reuse
neutral_output = model()
v_template = neutral_output.v_shaped

# Reuse for multiple poses
for pose in pose_batch:
    output = model(body_pose=pose)
```

### GPU Acceleration
```python
# Move model to GPU
device = torch.device('cuda')
model = model.to(device)

# Ensure inputs are on same device
betas = betas.to(device)
output = model(betas=betas)
```

## References

- **Paper**: [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de)
- **GitHub**: https://github.com/vchoutas/smplx
- **Documentation**: https://github.com/vchoutas/smplx/tree/main/transfer_model
- **Model Downloads**: Register at https://smpl-x.is.tue.mpg.de

## Citation

```bibtex
@inproceedings{SMPL-X:2019,
    title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
    author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and 
              Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and 
              Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```