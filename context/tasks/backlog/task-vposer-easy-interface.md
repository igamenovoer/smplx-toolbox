# Metadata

- Date: 2025-09-15
- Branch: main
- Commit: 209157a6e6589b311b351ba68c246702038d051e

# How to add easy inference helpers for VPoserModel (easy_encode/easy_decode)

This guide shows how to provide user-friendly inference methods that “just work” with tensors or NumPy inputs, on whatever device the module currently lives on, without changing your core `encode`/`decode` implementations.

Goals
- Keep `encode`/`decode` pure (no hidden device/dtype moves/casts inside them).
- Add convenience wrappers (e.g., `easy_encode`, `easy_decode`) that:
    - Accept `torch.Tensor` or `numpy.ndarray`.
  - Infer model device/dtype and move inputs accordingly.
  - Use `torch.inference_mode()` for inference-only speedups.
  - Optionally use mixed precision (`torch.autocast`) on CUDA for speed.
  - Avoid hardcoding devices (never assume `"cuda:0"`).

Recommended design
1) Leave `encode` and `decode` unchanged. They assume inputs are already on the right device/dtype.
2) Implement the “easy API” as separate methods (or a small decorator) that handle:
    - Input normalization (Tensor/NumPy → Tensor)
   - Device placement to the model’s device
   - Sensible dtype policy (avoid surprise up/downcasts; fix only float64→float32 by default)
   - Inference contexts: `torch.inference_mode()` and optional CUDA AMP

Minimal building blocks
```python
import torch, numpy as np
from typing import Any, Tuple

def _model_device_dtype(m: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(m.parameters(), None)
    if p is not None:
        return p.device, p.dtype
    b = next(m.buffers(), None)
    if b is not None:
        return b.device, b.dtype
    return torch.device("cpu"), torch.float32  # fallback for parameterless modules

def _to_tensor(x: Any, copy_numpy: bool = False) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, copy=copy_numpy)
    # best-effort fallback (lists, scalars, etc.)
    return torch.as_tensor(x, copy=True)
```

easy_encode (inference-only)
```python
def easy_encode(
    self, x: Any,
    *,
    precision: str = "auto",   # {'auto','fp32','fp16','bf16'}
    fix_float64: bool = True,   # downcast float64 → float32 once
    non_blocking: bool = True,
    return_stats: bool = False, # if True, also return (mu, std)
):
    """User-friendly encoder wrapper.

    Accepts Tensor/NumPy, moves to the module's device, applies
    inference_mode and optional AMP, then calls self.encode(x).
    """
    device, _ = _model_device_dtype(self)
    t = _to_tensor(x)
    if fix_float64 and t.dtype == torch.float64:
        t = t.to(dtype=torch.float32)
    t = t.to(device=device, non_blocking=non_blocking)

    # decide AMP policy
    amp_enabled = False
    amp_dtype = None
    if precision == "fp32":
        amp_enabled = False
    elif precision in ("fp16", "float16"):
        amp_enabled = device.type == "cuda"; amp_dtype = torch.float16
    elif precision in ("bf16", "bfloat16"):
        if device.type in ("cuda", "cpu"):
            amp_enabled = True; amp_dtype = torch.bfloat16
    else:  # 'auto'
        if device.type == "cuda":
            bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            amp_enabled = True; amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    self.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            q_z = self.encode(t)  # VPoserModel.encode returns torch.distributions.Normal
            # For inference convenience, return mean sample by default
            z = q_z.mean  # or q_z.rsample() if you want stochastic behavior
            if return_stats:
                return z, (q_z.mean, q_z.scale)
            return z
```

easy_decode (inference-only)
```python
def easy_decode(
    self, z: Any,
    *,
    precision: str = "auto",
    fix_float64: bool = True,
    non_blocking: bool = True,
    return_numpy: bool = False,
):
    """User-friendly decoder wrapper.

    Accepts Tensor/NumPy latent, moves to module's device, applies
    inference_mode and optional AMP, then calls self.decode(z).
    """
    device, _ = _model_device_dtype(self)
    t = _to_tensor(z)
    if fix_float64 and t.dtype == torch.float64:
        t = t.to(dtype=torch.float32)
    t = t.to(device=device, non_blocking=non_blocking)

    amp_enabled = False
    amp_dtype = None
    if precision == "fp32":
        amp_enabled = False
    elif precision in ("fp16", "float16"):
        amp_enabled = device.type == "cuda"; amp_dtype = torch.float16
    elif precision in ("bf16", "bfloat16"):
        if device.type in ("cuda", "cpu"):
            amp_enabled = True; amp_dtype = torch.bfloat16
    else:
        if device.type == "cuda":
            bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            amp_enabled = True; amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    self.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = self.decode(t)  # expects {'pose_body': (B,21,3), 'pose_body_matrot': (B,21,9)}

    if return_numpy:
        def to_np(v):
            if torch.is_tensor(v):
                return v.detach().to("cpu").numpy()
            return v
        return {k: to_np(v) for k, v in out.items()}
    return out
```

Usage examples
```python
vae = VPoserModel(cfg)
vae.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Encode a NumPy body pose of shape (B, 63) or (B, 21, 3)
pose_np = np.random.randn(2, 63)
z = easy_encode(vae, pose_np)               # returns mean latent (B, D)
z16 = easy_encode(vae, pose_np, precision="fp16")
z32, (mu, std) = easy_encode(vae, pose_np, return_stats=True)

# Decode a latent (Tensor/NumPy)
decoded = easy_decode(vae, z)
decoded_np = easy_decode(vae, z, return_numpy=True)
```

Why this pattern
- Separation of concerns: core math stays pure; user ergonomics live in the wrappers.
- Performance: device moves happen once at the boundary; AMP is optional and measured.
- Predictability: no hidden device/dtype mutations inside `encode`/`decode`.

Common pitfalls and tips
- Do not hardcode `"cuda:0"`. Always infer `device` from model params/buffers.
- Avoid blanket `.half()`/`.bfloat16()` inside core methods. Prefer AMP contexts.
- NumPy defaults to `float64`; downcast to `float32` once to avoid slow doubles.
- Register constants as buffers so they track device/dtype with `model.to(...)`.
- If inputs can be nested structures (dict/list/tuple), extend `_to_tensor` and device move with a recursive map.
- For throughput, consider `DataLoader(pin_memory=True)` and `to(..., non_blocking=True)` on CUDA.

Related alternatives
- PyTorch Lightning: `@auto_move_data` decorator moves method inputs to module device automatically. Combine with `@torch.inference_mode()` for inference helpers.
- fastai: `to_device` utility recursively moves nested structures to a target device.
- Hugging Face Accelerate: manages device placement and autocast globally via an `Accelerator` object.

References
- PyTorch AMP: Automatic Mixed Precision — https://docs.pytorch.org/docs/stable/amp.html
- `torch.inference_mode()` — https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
- Lightning `auto_move_data` decorator — https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#auto-move-data
