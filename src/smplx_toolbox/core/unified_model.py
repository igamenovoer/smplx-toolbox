"""Unified SMPL Family Model - provides a unified API for SMPL, SMPL-H and SMPL-X models.

This module provides a single Python class that abstracts model differences
(SMPL vs SMPL-H vs SMPL-X) while exposing a consistent interface for posing
and retrieving outputs.

Key Components:
    - UnifiedSmplModel: Main adapter class wrapping smplx models.
    - UnifiedSmplInputs: Standardized input container.
    - PoseByKeypoints: User-friendly per-joint pose specification.
    - UnifiedSmplOutput: Standardized output container.

Usage Pattern
-------------
.. code-block:: python

    import torch
    import smplx
    from smplx_toolbox.core.unified_model import UnifiedSmplModel, PoseByKeypoints

    # 1. Load a base SMPL-family model (e.g., SMPL-X)
    base_smplx_model = smplx.create(
        model_path='/path/to/smpl/models',
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        batch_size=1
    )

    # 2. Wrap it with the UnifiedSmplModel
    model = UnifiedSmplModel.from_smpl_model(base_smplx_model)

    # 3. Define a pose using the user-friendly PoseByKeypoints class
    #    (all poses are in axis-angle format)
    keypoint_pose = PoseByKeypoints(
        left_shoulder=torch.tensor([[0.0, 0.0, -1.5]]),  # Raise left arm
        right_shoulder=torch.tensor([[0.0, 0.0, 1.5]]),   # Raise right arm
        jaw=torch.tensor([[0.2, 0.0, 0.0]])             # Open jaw (SMPL-X only)
    )

    # 4. Run the forward pass
    #    The wrapper automatically converts keypoint poses to the format
    #    expected by the base model.
    output = model(keypoint_pose)

    # 5. Access unified outputs
    print(f"Model type: {model.model_type}")
    print(f"Vertices shape: {output.vertices.shape}")
    print(f"Unified joints shape: {output.joints.shape}")

    # Access specific joint groups
    body_joints = output.body_joints
    hand_joints = output.hand_joints

    # Select joints by name from the unified set
    shoulders = model.select_joints(output.joints, names=["left_shoulder", "right_shoulder"])
    print(f"Shoulder positions shape: {shoulders.shape}")

"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Import from our sub-modules
from .constants import (
    DeviceLike,
    MissingJointFill,
    ModelType,
    T,
    get_smpl_joint_names,
    get_smplh_joint_names,
    get_smplx_joint_names,
)
from .containers import PoseByKeypoints, UnifiedSmplInputs, UnifiedSmplOutput

__all__ = [
    "UnifiedSmplModel"
]


class UnifiedSmplModel:
    """Unified adapter for SMPL family models (SMPL, SMPL-H, SMPL-X).

    This class provides a consistent interface for working with different SMPL
    model variants, abstracting away their differences while exposing common
    functionality.

    Key features:
        - Auto-detection of model type from the provided `smplx` model instance.
        - Normalized inputs and outputs across model variants.
        - Unification of joint sets to the 55-joint SMPL-X scheme.
        - Support for user-friendly per-keypoint pose specification via `PoseByKeypoints`.

    Note
    ----
    Use the `from_smpl_model` classmethod to create configured instances.
    """

    def __init__(self) -> None:
        """Initialize an empty wrapper.

        Use the `from_smpl_model` classmethod to create a configured instance.
        """
        self.m_deformable_model: nn.Module | None = None
        self.m_missing_joint_fill: MissingJointFill | None = None
        self.m_warn_fn: Callable[[str], None] | None = None
        # Optional auxiliary mapping tensors (lazy initialized)
        self.m_joint_mapping: dict[int, int] | None = None

    @classmethod
    def from_smpl_model(
        cls: type[T],
        deformable_model: nn.Module,
        *,
        missing_joint_fill: MissingJointFill | str = MissingJointFill.NAN,
        warn_fn: Callable[[str], None] | None = None
    ) -> T:
        """Create a unified model wrapper from an existing SMPL-family model instance.

        Parameters
        ----------
        deformable_model : torch.nn.Module
            A pre-loaded model instance, typically created using `smplx.create`.
        missing_joint_fill : {'nan', 'zero'}, optional
            Specifies how to fill joint positions that are part of the unified
            55-joint set but not present in the base model. Defaults to 'nan'.
        warn_fn : Callable[[str], None], optional
            A custom function to handle warnings. Defaults to `warnings.warn`.

        Returns
        -------
        T
            A configured instance of the `UnifiedSmplModel`.
        """
        instance = cls()
        instance.m_deformable_model = deformable_model
        # Normalize missing joint fill strategy to enum
        if isinstance(missing_joint_fill, MissingJointFill):
            instance.m_missing_joint_fill = missing_joint_fill
        else:
            # Accept string values for backward compatibility
            try:
                instance.m_missing_joint_fill = MissingJointFill(missing_joint_fill)
            except ValueError:
                raise ValueError(
                    f"Invalid missing_joint_fill: {missing_joint_fill}. "
                    f"Expected one of: {MissingJointFill.values()}"
                )
        instance.m_warn_fn = warn_fn or warnings.warn

        return instance

    def _detect_model_type(self) -> ModelType:
        """Auto-detect the model type from the wrapped `smplx` instance."""
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")

        model: Any = self.m_deformable_model
        type_name = type(model).__name__.lower()

        # Try type name first
        if type_name in ModelType.values():
            return ModelType(type_name)

        # Heuristics based on attributes
        if hasattr(model, "jaw_pose") or hasattr(model, "leye_pose") or hasattr(model, "reye_pose"):
            return ModelType.SMPLX
        elif hasattr(model, "left_hand_pose") and hasattr(model, "right_hand_pose"):
            return ModelType.SMPLH
        else:
            return ModelType.SMPL

    @property
    def model_type(self) -> ModelType:
        """Get the detected model type of the wrapped model.

        Returns
        -------
        ModelType
            The model type value (stringy enum: compares equal to 'smpl'/'smplh'/'smplx').
        """
        return self._detect_model_type()

    @property
    def num_betas(self) -> int:
        """Get the number of shape parameters (betas) of the wrapped model."""
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")
        model: Any = self.m_deformable_model
        if hasattr(model, "num_betas"):
            return int(model.num_betas)
        elif hasattr(model, "shapedirs"):
            return model.shapedirs.shape[-1]
        elif hasattr(model, "betas") and model.betas is not None:
            return model.betas.shape[-1]
        return 10  # Default

    @property
    def num_expressions(self) -> int:
        """Get the number of expression parameters.

        Returns
        -------
        int
            The number of expression parameters. Returns 0 for non-SMPL-X models.
        """
        if self.model_type != "smplx":
            return 0
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")
        model: Any = self.m_deformable_model
        if hasattr(model, "num_expression_coeffs"):
            return int(model.num_expression_coeffs)
        elif hasattr(model, "expression") and model.expression is not None:
            return model.expression.shape[-1]
        return 10  # Default for SMPL-X

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the wrapped model's parameters."""
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")
        model: Any = self.m_deformable_model
        if hasattr(model, "v_template"):
            return model.v_template.dtype
        elif hasattr(model, "shapedirs"):
            return model.shapedirs.dtype
        return torch.float32

    @property
    def device(self) -> torch.device:
        """Get the device of the wrapped model's parameters."""
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")
        model: Any = self.m_deformable_model
        # Try parameters first
        try:
            param = next(model.parameters())
            return param.device
        except StopIteration:
            pass
        # Try buffers
        try:
            buffer = next(model.buffers())
            return buffer.device
        except StopIteration:
            pass
        return torch.device("cpu")

    @property
    def faces(self) -> Tensor:
        """Get the face connectivity tensor from the wrapped model.

        Returns
        -------
        torch.Tensor
            A `torch.long` tensor of face indices, typically on the CPU.

        Notes
        -----
        Faces are usually kept on the CPU as they are primarily used for
        rendering and visualization rather than for computation in the forward pass.
        """
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")
        model: Any = self.m_deformable_model
        if hasattr(model, "faces_tensor"):
            faces = model.faces_tensor
            # Ensure torch.long dtype
            if faces.dtype != torch.long:
                return faces.long()  # type: ignore
            return faces
        elif hasattr(model, "faces"):
            faces = model.faces
            if isinstance(faces, np.ndarray):
                return torch.from_numpy(faces).long()
            # Ensure torch.long dtype for tensors
            if faces.dtype != torch.long:
                return faces.long()  # type: ignore
            return faces
        raise AttributeError("Model has no faces")

    def _normalize_inputs(self, inputs: UnifiedSmplInputs | PoseByKeypoints) -> dict[str, Tensor]:
        """Normalize and prepare inputs for the wrapped `smplx` model.

        This method handles the conversion from `UnifiedSmplInputs` or
        `PoseByKeypoints` to the dictionary of keyword arguments expected by
        the underlying `smplx` model's forward pass. It ensures all required
        tensors are present, correctly shaped, and on the correct device.

        Parameters
        ----------
        inputs : UnifiedSmplInputs or PoseByKeypoints
            The high-level input specification.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of normalized tensor parameters ready to be passed to
            the `smplx` model.
        """
        model_type = self.model_type

        # Convert keypoints if needed
        if isinstance(inputs, PoseByKeypoints):
            inputs.check_valid_by_keypoints(model_type, strict=False, warn_fn=self.m_warn_fn)
            inputs = UnifiedSmplInputs.from_keypoint_pose(inputs, model_type=model_type)

        # Validate inputs (beta/expr count adapted later)
        inputs.check_valid(model_type, num_betas=None, num_expressions=self.num_expressions)

        # Route through input conversion helpers (AA-space)
        if model_type == "smpl":
            input_dict: dict[str, Tensor | bool] = inputs.to_smpl_inputs()
        elif model_type == "smplh":
            input_dict = inputs.to_smplh_inputs(with_hand_shape=False)
        else:
            input_dict = inputs.to_smplx_inputs()

        batch_size = inputs.batch_size() or 1
        device = self.device
        dtype = self.dtype

        # Helper to ensure tensor or create zeros
        def ensure(value: Tensor | None, shape: tuple[int, ...]) -> Tensor:
            if value is not None:
                return value.to(device=device, dtype=dtype)
            return torch.zeros(shape, device=device, dtype=dtype)

        normalized: dict[str, Tensor | bool] = {}

        # Global orient
        normalized["global_orient"] = ensure(input_dict.get("global_orient", None), (batch_size, 3))

        # Body pose: pad to 69 for SMPL
        body_pose_63 = ensure(input_dict.get("body_pose", None), (batch_size, 63))
        if model_type == "smpl":
            pad = torch.zeros((batch_size, 6), device=device, dtype=dtype)
            normalized["body_pose"] = torch.cat([body_pose_63, pad], dim=-1)
        else:
            normalized["body_pose"] = body_pose_63

        # Betas: adapt to model's expected count by pad/truncate
        if inputs.betas is not None:
            b = inputs.betas.to(device=device, dtype=dtype)
            nb = self.num_betas
            if b.shape[1] < nb:
                pad_b = torch.zeros((b.shape[0], nb - b.shape[1]), device=device, dtype=dtype)
                b = torch.cat([b, pad_b], dim=1)
            elif b.shape[1] > nb:
                if self.m_warn_fn:
                    self.m_warn_fn(
                        f"Truncating betas from {b.shape[1]} to {nb} to match model"
                    )
                b = b[:, :nb]
            normalized["betas"] = b

        # Translation
        if inputs.trans is not None:
            normalized["transl"] = inputs.trans.to(device=device, dtype=dtype)

        # Hands (handle PCA models)
        if model_type in ["smplh", "smplx"]:
            model = self.m_deformable_model  # type: ignore[assignment]
            use_pca = bool(getattr(model, "use_pca", False))
            if use_pca:
                # Convert AA(45) -> PCA(K) using lstsq against components; subtract means
                lh_aa = input_dict.get("left_hand_pose", None)
                rh_aa = input_dict.get("right_hand_pose", None)
                lh_comp = getattr(model, "left_hand_components", None)
                rh_comp = getattr(model, "right_hand_components", None)
                lh_mean = getattr(model, "left_hand_mean", None)
                rh_mean = getattr(model, "right_hand_mean", None)

                def aa_to_pca(aa: Tensor | None, comp: Tensor | None, mean: Tensor | None) -> Tensor:
                    K = int(getattr(model, "num_pca_comps", 6))
                    if aa is None:
                        return torch.zeros((batch_size, K), device=device, dtype=dtype)
                    if comp is None:
                        if self.m_warn_fn:
                            self.m_warn_fn("Missing hand PCA components; using zeros for coefficients")
                        return torch.zeros((aa.shape[0], K), device=device, dtype=dtype)
                    # Shapes: comp: (K,45), aa: (B,45), mean: (45,)
                    aa_tgt = aa.to(device=device, dtype=dtype)
                    if mean is not None:
                        mean_t = mean.to(device=device, dtype=dtype)
                        if mean_t.ndim == 1:
                            mean_t = mean_t.view(1, -1)
                        aa_tgt = aa_tgt - mean_t
                    # Solve comp^T @ X^T ≈ aa_tgt^T  => (45,K) @ (K,B) = (45,B)
                    A = comp.t().to(device=device, dtype=dtype)  # (45,K)
                    Y = aa_tgt.t()  # (45,B)
                    try:
                        sol = torch.linalg.lstsq(A, Y).solution  # (K,B)
                    except Exception:
                        pinv = torch.linalg.pinv(A)
                        sol = pinv @ Y
                    return sol.t().contiguous()  # (B,K)

                normalized["left_hand_pose"] = aa_to_pca(lh_aa, lh_comp, lh_mean)
                normalized["right_hand_pose"] = aa_to_pca(rh_aa, rh_comp, rh_mean)
            else:
                # Pass AA(45) directly
                normalized["left_hand_pose"] = ensure(input_dict.get("left_hand_pose", None), (batch_size, 45))
                normalized["right_hand_pose"] = ensure(input_dict.get("right_hand_pose", None), (batch_size, 45))

        # SMPL-X specifics (jaw/eyes/expressions)
        if model_type == "smplx":
            normalized["jaw_pose"] = ensure(input_dict.get("jaw_pose", None), (batch_size, 3))
            normalized["leye_pose"] = ensure(input_dict.get("leye_pose", None), (batch_size, 3))
            normalized["reye_pose"] = ensure(input_dict.get("reye_pose", None), (batch_size, 3))

            num_expr = self.num_expressions
            if num_expr > 0:
                expr_in = input_dict.get("expression", None)
                if expr_in is None:
                    normalized["expression"] = torch.zeros((batch_size, num_expr), device=device, dtype=dtype)
                else:
                    e = expr_in.to(device=device, dtype=dtype)
                    if e.shape[1] < num_expr:
                        pad_e = torch.zeros((e.shape[0], num_expr - e.shape[1]), device=device, dtype=dtype)
                        e = torch.cat([e, pad_e], dim=1)
                    elif e.shape[1] > num_expr:
                        if self.m_warn_fn:
                            self.m_warn_fn(
                                f"Truncating expression from {e.shape[1]} to {num_expr} to match model"
                            )
                        e = e[:, :num_expr]
                    normalized["expression"] = e

        # Always request vertices
        normalized["return_verts"] = True

        return normalized  # type: ignore[return-value]

    def _unify_joints(self, joints_raw: Tensor, model_type: ModelType | str) -> tuple[Tensor, dict[str, Any]]:
        """Convert raw model joints to the unified 55-joint SMPL-X set.

        This method maps the joint output from any SMPL-family model to the
        standard 55-joint SMPL-X format, filling in missing joints (e.g., face
        joints for SMPL/SMPL-H) as specified by `missing_joint_fill`.

        Parameters
        ----------
        joints_raw : torch.Tensor
            The raw joint positions from the base model, of shape (B, J_raw, 3).
        model_type : str
            The source model type ('smpl', 'smplh', 'smplx').

        Returns
        -------
        tuple[torch.Tensor, dict[str, Any]]
            A tuple containing:
            - The unified joints tensor of shape (B, 55, 3).
            - An `extras` dictionary with mapping information and the raw joints.
        """
        batch_size = joints_raw.shape[0]
        device = joints_raw.device
        dtype = joints_raw.dtype

        extras: dict[str, Any] = {
            "joints_raw": joints_raw,
            "joint_mapping": {},  # raw_idx -> unified_idx
            "joint_names_raw": self._get_raw_joint_names()
        }

        if model_type == "smplx":
            # SMPL-X: direct mapping (first 55 are the standard set)
            joints_unified = joints_raw[:, :55]
            # Create identity mapping
            for i in range(55):
                extras["joint_mapping"][i] = i

        elif model_type == "smplh":
            # SMPL-H: has body (22) + hands (30) = 52 joints
            # Add placeholders for face joints (jaw, left_eye, right_eye)
            joints_unified = torch.zeros((batch_size, 55, 3), device=device, dtype=dtype)

            # Map body joints (0-21 -> 0-21)
            joints_unified[:, :22] = joints_raw[:, :22]
            for i in range(22):
                extras["joint_mapping"][i] = i

            # Map hand joints (22-51 -> 25-54)
            joints_unified[:, 25:55] = joints_raw[:, 22:52]
            for i in range(30):
                extras["joint_mapping"][22 + i] = 25 + i

            # Fill missing face joints (indices 22, 23, 24: jaw, left_eye_smplhf, right_eye_smplhf)
            if self.m_missing_joint_fill == MissingJointFill.NAN:
                joints_unified[:, 22:25] = float("nan")
            # else: already zeros

            extras["missing_joints"] = [22, 23, 24]  # jaw, left_eye, right_eye

        else:  # smpl
            # SMPL: Build mapping based on joint names
            joints_unified = torch.zeros((batch_size, 55, 3), device=device, dtype=dtype)

            # Create name-based mapping
            raw_names = get_smpl_joint_names()[:joints_raw.shape[1]]
            unified_names = get_smplx_joint_names()[:55]

            for raw_idx, raw_name in enumerate(raw_names):
                # Find corresponding unified index
                if raw_name in unified_names:
                    unified_idx = unified_names.index(raw_name)
                    joints_unified[:, unified_idx] = joints_raw[:, raw_idx]
                    extras["joint_mapping"][raw_idx] = unified_idx

            # Track missing joints (hands and face)
            missing = []
            for i in range(55):
                if i not in extras["joint_mapping"].values():
                    missing.append(i)

            # Fill missing joints
            if self.m_missing_joint_fill == MissingJointFill.NAN:
                for idx in missing:
                    joints_unified[:, idx] = float("nan")
            # else: already zeros

            extras["missing_joints"] = missing

        return joints_unified, extras

    def _get_raw_joint_names(self) -> list[str] | None:
        """Get the raw, model-specific joint names from the wrapped model."""
        if self.m_deformable_model is None:
            return None

        if hasattr(self.m_deformable_model, "joint_names"):
            return self.m_deformable_model.joint_names  # type: ignore

        # Use default names based on model type
        model_type = self.model_type
        if model_type == "smplx":
            return get_smplx_joint_names()
        elif model_type == "smplh":
            return get_smplh_joint_names()
        elif model_type == "smpl":
            return get_smpl_joint_names()

        return None

    def _compute_full_pose(self, normalized_inputs: dict[str, Tensor]) -> Tensor:
        """Compute the full flattened pose vector used for Linear Blend Skinning (LBS).

        This method concatenates all pose parameters (root, body, hands, face)
        in the correct order required by the underlying `smplx` model.

        Parameters
        ----------
        normalized_inputs : dict[str, torch.Tensor]
            The dictionary of normalized inputs for the model.

        Returns
        -------
        torch.Tensor
            The full, flattened pose vector of shape (B, P), where P is the
            total number of pose parameters for the model type.
        """
        model_type = self.model_type
        pose_parts = []

        # Always start with root
        if "global_orient" in normalized_inputs:
            pose_parts.append(normalized_inputs["global_orient"])

        # Body pose
        if "body_pose" in normalized_inputs:
            bp = normalized_inputs["body_pose"]
            # Report 63-DoF body pose in the unified full_pose for SMPL/SMPL-H,
            # even if we padded to 69 for the model call.
            if self.model_type in ["smpl", "smplh"] and bp.shape[-1] >= 63:
                pose_parts.append(bp[:, :63])
            else:
                pose_parts.append(bp)

        # Model-specific parts
        if model_type == "smplx":
            # SMPL-X: add jaw and eyes before hands
            if "jaw_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["jaw_pose"])
            if "leye_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["leye_pose"])
            if "reye_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["reye_pose"])

        # Hands (SMPL-H and SMPL-X)
        if model_type in ["smplh", "smplx"]:
            # If the underlying model uses PCA for hands (e.g., 6 comps), expand to 45 AA for reporting
            model = self.m_deformable_model  # type: ignore[assignment]
            use_pca = bool(getattr(model, "use_pca", False))
            if use_pca:
                # Try to expand PCA -> axis-angle using model's components buffers
                lh = normalized_inputs.get("left_hand_pose")
                rh = normalized_inputs.get("right_hand_pose")
                lh_comp = getattr(model, "left_hand_components", None)
                rh_comp = getattr(model, "right_hand_components", None)
                device, dtype = self.device, self.dtype

                def expand_pca(p: Tensor | None, comp: Tensor | None) -> Tensor:
                    if p is None:
                        return torch.zeros((1, 45), device=device, dtype=dtype)
                    if comp is None:
                        # Fallback to zeros with AA size if components unavailable
                        return torch.zeros((p.shape[0], 45), device=p.device, dtype=p.dtype)
                    # Ensure comp is on same device/dtype
                    comp_t = comp.to(device=p.device, dtype=p.dtype)
                    # bi,ij->bj : (B, K) @ (K, 45) => (B, 45)
                    return torch.einsum("bi,ij->bj", p, comp_t)

                if lh is not None:
                    pose_parts.append(expand_pca(lh, lh_comp))
                if rh is not None:
                    pose_parts.append(expand_pca(rh, rh_comp))
            else:
                if "left_hand_pose" in normalized_inputs:
                    pose_parts.append(normalized_inputs["left_hand_pose"])
                if "right_hand_pose" in normalized_inputs:
                    pose_parts.append(normalized_inputs["right_hand_pose"])

        if pose_parts:
            return torch.cat(pose_parts, dim=-1)
        else:
            # Fallback: return empty pose
            batch_size = 1
            if "global_orient" in normalized_inputs:
                batch_size = normalized_inputs["global_orient"].shape[0]
            return torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)

    def __call__(self, inputs: UnifiedSmplInputs | PoseByKeypoints) -> UnifiedSmplOutput:
        """Forward pass through the model. See `forward` for details."""
        return self.forward(inputs)

    def forward(self, inputs: UnifiedSmplInputs | PoseByKeypoints) -> UnifiedSmplOutput:
        """Run a forward pass through the unified model.

        This method takes either a `UnifiedSmplInputs` or a `PoseByKeypoints`
        object, normalizes the inputs, passes them to the wrapped `smplx` model,
        and returns the results in a standardized `UnifiedSmplOutput` container.

        Parameters
        ----------
        inputs : UnifiedSmplInputs or PoseByKeypoints
            The input specification for the model.

        Returns
        -------
        UnifiedSmplOutput
            A standardized container with the model's output, including vertices,
            faces, and unified joints.
        """
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")

        # Normalize inputs
        normalized = self._normalize_inputs(inputs)

        # Call wrapped model
        output = self.m_deformable_model(**normalized)

        # Extract outputs
        vertices = output.vertices
        joints_raw = output.joints

        # Unify joints
        joints_unified, extras = self._unify_joints(joints_raw, self.model_type)

        # Compute full pose
        full_pose = self._compute_full_pose(normalized)

        # Add any extra outputs
        if hasattr(output, "v_shaped"):
            extras["v_shaped"] = output.v_shaped
        # Expose input betas in extras for priors/losses
        if "betas" in normalized:
            betas_val = normalized["betas"]
            if isinstance(betas_val, torch.Tensor):
                extras["betas"] = betas_val

        return UnifiedSmplOutput(
            vertices=vertices,
            faces=self.faces,
            joints=joints_unified,
            full_pose=full_pose,
            extras=extras
        )

    def to(self, device: DeviceLike) -> UnifiedSmplModel:  # noqa: ARG002
        """Move the model's auxiliary tensors to the specified device.

        .. warning::
            This method only moves tensors owned by the `UnifiedSmplModel` adapter
            itself (if any). You must still move the wrapped `smplx` model to the
            device separately, e.g., `model.m_deformable_model.to(device)`.

        Parameters
        ----------
        device : DeviceLike
            The target device (e.g., 'cuda:0' or `torch.device('cpu')`).

        Returns
        -------
        UnifiedSmplModel
            The model instance for chaining.
        """
        # Currently no auxiliary tensors to move
        # Future: move any cached mapping tensors
        if self.m_warn_fn:
            self.m_warn_fn(
                "UnifiedSmplModel.to() only moves adapter tensors. "
                "Move the wrapped model with model.m_deformable_model.to(device)"
            )
        return self

    def eval(self) -> UnifiedSmplModel:
        """Set the wrapped model to evaluation mode."""
        if self.m_deformable_model is not None:
            self.m_deformable_model.eval()
        return self

    def train(self, mode: bool = True) -> UnifiedSmplModel:
        """Set the wrapped model to training mode."""
        if self.m_deformable_model is not None:
            self.m_deformable_model.train(mode)
        return self

    def get_joint_names(self, unified: bool = True) -> list[str]:
        """Get the list of joint names for the model.

        Parameters
        ----------
        unified : bool, optional
            If True (default), return the names for the 55-joint unified SMPL-X
            set. If False, return the raw, model-specific joint names.

        Returns
        -------
        list[str]
            A list of joint names.
        """
        if unified:
            # Return the first 55 official SMPL-X joint names
            return get_smplx_joint_names()[:55]
        else:
            # Return model-specific names if available
            names = self._get_raw_joint_names()
            if names:
                return names
            return [f"raw_joint_{i}" for i in range(self._get_raw_joint_count())]

    def _get_raw_joint_count(self) -> int:
        """Get the raw, model-specific joint count."""
        model_type = self.model_type
        if model_type == "smplx":
            return 55
        elif model_type == "smplh":
            return 52
        else:  # smpl
            return 24

    def select_joints(
        self,
        joints: Tensor,
        indices: list[int] | Tensor | None = None,
        names: list[str] | None = None
    ) -> Tensor:
        """Select a subset of joints by their indices or names.

        Parameters
        ----------
        joints : torch.Tensor
            A joint tensor of shape (B, J, 3), typically from a `UnifiedSmplOutput`.
        indices : list[int] or torch.Tensor, optional
            A list or tensor of integer indices specifying which joints to select.
        names : list[str], optional
            A list of joint names to select. The names are mapped to indices
            using the unified 55-joint set.

        Returns
        -------
        torch.Tensor
            A tensor of shape (B, n, 3) containing only the selected joints,
            where n is the number of specified indices or names.

        Raises
        ------
        ValueError
            If both `indices` and `names` are provided.
        """
        if indices is not None and names is not None:
            raise ValueError("Provide either indices or names, not both")

        if names is not None:
            # Convert names to indices
            joint_names = self.get_joint_names(unified=True)
            indices = []
            for name in names:
                if name in joint_names:
                    indices.append(joint_names.index(name))
                else:
                    if self.m_warn_fn:
                        self.m_warn_fn(f"Joint name '{name}' not found")

        if indices is not None:
            if isinstance(indices, list):
                indices = torch.tensor(indices, dtype=torch.long, device=joints.device)
            return joints[:, indices]

        return joints
