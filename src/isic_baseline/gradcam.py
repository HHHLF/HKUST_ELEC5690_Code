"""Grad-CAM on DINOv2 patch tokens.

The classification head reads only the final CLS token. Patch outputs of the same
last block do not affect that CLS token, so their gradients are zero. The hook is
placed on ``blocks[-1].norm1``, whose patch tokens are keys/values for the CLS
query. The CLS position is removed and the remaining tokens are reshaped to the
patch grid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ViTGradCAM:
    def __init__(self, model, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, image: torch.Tensor, class_index: int) -> np.ndarray:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Grad-CAM expects a batch of one image.")
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        logits = self.model(image)
        if class_index < 0 or class_index >= logits.shape[1]:
            raise IndexError(f"class_index {class_index} outside logits {tuple(logits.shape)}")
        score = logits[0, class_index]
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activation and gradient.")
        activations = self.activations[0]
        gradients = self.gradients[0]
        if activations.shape != gradients.shape or activations.ndim != 2:
            raise RuntimeError(f"Unexpected token tensors: act {tuple(activations.shape)} grad {tuple(gradients.shape)}")
        # Token 0 is CLS. It is not a spatial location.
        patch_act = activations[1:]
        patch_grad = gradients[1:]
        grid = int(round(patch_act.shape[0] ** 0.5))
        if grid * grid != patch_act.shape[0]:
            raise RuntimeError(f"Patch token count {patch_act.shape[0]} is not a square grid.")
        weights = patch_grad.mean(dim=0)
        cam = torch.relu((patch_act * weights).sum(dim=-1))
        cam = cam.reshape(1, 1, grid, grid)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        flat = cam.detach()
        if float(flat.abs().sum()) <= 0 or float(flat.std()) <= 0:
            raise RuntimeError("Grad-CAM map is all zeros or constant, so this target layer is not usable.")
        cam = cam / flat.amax().clamp_min(1e-8)
        return cam.detach().cpu().numpy().astype(np.float32)


def generate_gradcam(model, image: torch.Tensor, class_index: int) -> tuple[np.ndarray, str]:
    """Try target layers that actually influence the CLS logit, and reject empty maps."""
    candidates = [
        ("backbone.blocks[-1].norm1", model.backbone.blocks[-1].norm1),
        ("backbone.blocks[-2]", model.backbone.blocks[-2]),
    ]
    errors = []
    for name, layer in candidates:
        cam_engine = ViTGradCAM(model, layer)
        try:
            heatmap = cam_engine(image, class_index)
            return heatmap, name
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
        finally:
            cam_engine.close()
    raise RuntimeError("No Grad-CAM target produced a non-empty map. " + " | ".join(errors))
