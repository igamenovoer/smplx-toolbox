# Unified SMPL model processing

## Model files
- SMPL: `data/body_models/smpl/SMPL_NEUTRAL.pkl`
- SMPL-H: `data/body_models/smplh/SMPLH_MALE.pkl`
- SMPL-H-mano: `data/body_models/mano_v1_2/models/SMPLH_male.pkl`
- SMPL-H (NPZ variant): `data/body_models/smplh/male/model.npz`
- SMPL-X (PKL): `data/body_models/smplx/SMPLX_MALE.pkl`
- SMPL-X (NPZ): `data/body_models/smplx/SMPLX_MALE.npz`

## Model relationships
- overview: `context/hints/smplx-kb/compare-smpl-models.md`
- skeleton: `context/hints/smplx-kb/compare-smpl-skeleton.md`
- shapedirs: `context/hints/smplx-kb/compare-smpl-shape.md`
- conversion: `context/hints/smplx-kb/about-smplx-transfer-model.md`

## Reference Source Code and Data
- `smplx` library: `context/refcode/smplx`, note that it is just for reference, to use `smplx` we have installed it as a dependency
- `mano` library: `data/body_models/mano_v1_2`, contains both model and source code for loading mano-related models, though `smplx` library can also load mano models (check that)

## Current Implementation

- unified model class: `src/smplx_toolbox/core/unified_model.py`

## Requirements

we need to revise our unified model class, such that:
- it is initialized with model loaded with `smplx` library (like current implementation), where the actual model may be `SMPL`, `SMPL-H`, `SMPL-H-mano`, `SMPL-X`, including their pkl and npz variants (handled by `smplx` library)
- for the forward() function that computes vertices by parameters, we will use `UnifiedSmplInputs` class to include all possible parameters for all models, where some parameters may be ignored by certain models
- `UnifiedSmplInputs` has several conversion functions, converting the unified inputs to dict ([str, tensor]) that can be used by the forward() function of the underlying `smplx` model.
  - `to_smpl_inputs()`: convert to inputs for `SMPL` model
  - `to_smplh_inputs(with_hand_shape: bool)`: convert to inputs for `SMPL-H` and `SMPL-H-mano` models, `with_hand_shape` indicates whether to include hand shape parameters (`SMPL-H-mano` can use hand shape parameters)
  - `to_smplx_inputs()`: convert to inputs for `SMPL-X`

> what is the `smplx` model?
> ```python
> # this is how smplx libary creates a model
> # context/refcode/smplx/body_models.py
> def create(
>     model_path: str,
>     model_type: str = 'smpl',
>     **kwargs
> ) -> Union[SMPL, SMPLH, SMPLX, MANO, FLAME]:
>     ''' Method for creating a model from a path and a model type
> 
>         Parameters
>         ----------
>         model_path: str
>             Either the path to the model you wish to load or a folder,
>             where each subfolder contains the differents types, i.e.:
>             model_path:
>             |
>             |-- smpl
>                 |-- SMPL_FEMALE
>                 |-- SMPL_NEUTRAL
>                 |-- SMPL_MALE
>             |-- smplh
>                 |-- SMPLH_FEMALE
>                 |-- SMPLH_MALE
>             |-- smplx
>                 |-- SMPLX_FEMALE
>                 |-- SMPLX_NEUTRAL
>                 |-- SMPLX_MALE
>             |-- mano
>                 |-- MANO RIGHT
>                 |-- MANO LEFT
> 
>         model_type: str, optional
>             When model_path is a folder, then this parameter specifies  the
>             type of model to be loaded
>         **kwargs: dict
>             Keyword arguments
> 
>         Returns
>         -------
>             body_model: nn.Module
>                 The PyTorch module that implements the corresponding body model
>         Raises
>         ------
>             ValueError: In case the model type is not one of SMPL, SMPLH,
>             SMPLX, MANO or FLAME
>     '''
> ```

- because these models have shared joint names, we use `PoseByKeypoints` class to represent the pose parameters, joint angles are stored in named variables, and if a model does not have certain joints, the corresponding angles will be ignored. Such `PoseByKeypoints` instance will be accepted by `UnifiedSmplInputs` to create the input for forward() function

- we do not need to convert the parameters between different models, but we need to make sure they work similarly in API level: they respond correctly to given shape and pose parameters.