# Test-Time Optimization (TTO) for per-scene adaptation.
#
# At inference, clones LoRA + Head weights and performs gradient descent
# on the BRDF render loss to adapt to the specific test scene.
#
# Only updates: LoRA weights, Inverse Heads, Light Token.
# Does NOT update: frozen backbone, DINOv2 encoder, original VGGT heads.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class TestTimeOptimizer:
    """Test-Time Optimization for per-scene adaptation.

    Clones the trainable subset of model weights and performs gradient
    descent on the BRDF render loss for a fixed number of iterations.

    Args:
        iterations: Number of TTO iterations (default: 50).
        lr: Learning rate for TTO (default: 7e-4, between 1e-3 and 5e-4).
        params_patterns: List of name patterns identifying which parameters
            to optimize. Parameters whose name contains any of these patterns
            will be updated.
    """

    def __init__(
        self,
        iterations: int = 50,
        lr: float = 7e-4,
        params_patterns: Optional[List[str]] = None,
    ):
        self.iterations = iterations
        self.lr = lr
        self.params_patterns = params_patterns or [
            "lora_frame_blocks",
            "lora_global_blocks",
            "inverse_heads",
            "light_token",
        ]

    @torch.enable_grad()
    def optimize(
        self,
        model: nn.Module,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        brdf_renderer=None,
    ) -> Dict[str, torch.Tensor]:
        """Perform test-time optimization on a batch of images.

        The model's trainable parameters (LoRA + Heads) are cloned and
        optimized, then restored after optimization completes.

        Args:
            model: The VGGT model (should be in eval mode).
            images: Input images [B, S, 3, H, W] in [0, 1].
            brdf_renderer: Optional BRDFRenderer instance. If None, will
                attempt to import and create one.

        Returns:
            Dict of optimized predictions.
        """
        if brdf_renderer is None:
            from vggt.heads.brdf_renderer import BRDFRenderer
            brdf_renderer = BRDFRenderer(render_downsample=2)

        # 1. Save original parameter states
        was_training = model.training
        original_states = {}
        original_requires_grad = {}
        params_to_optimize = []

        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            if self._should_optimize(name):
                original_states[name] = param.data.clone()
                param.requires_grad_(True)
                params_to_optimize.append(param)
            else:
                param.requires_grad_(False)

        if not params_to_optimize:
            # Nothing to optimize, return regular predictions
            with torch.no_grad():
                return model(images, query_points=query_points, _disable_tto=True)

        # 2. Setup optimizer
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)

        # Target images for render loss
        images_hwc = images.permute(0, 1, 3, 4, 2).contiguous()

        try:
            # 3. Optimization loop
            model.train()  # Enable LoRA training path
            for _ in range(self.iterations):
                optimizer.zero_grad()

                # Forward pass
                predictions = model(images, query_points=query_points, _disable_tto=True)

                # Compute render loss if all required predictions are available
                loss = self._compute_tto_loss(predictions, images_hwc, brdf_renderer)

                if loss is None:
                    break

                if loss.requires_grad:
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                    optimizer.step()

            # 4. Final prediction with optimized weights
            model.eval()
            with torch.no_grad():
                final_predictions = model(images, query_points=query_points, _disable_tto=True)
        finally:
            # 5. Restore original weights and requires_grad flags
            for name, param in model.named_parameters():
                if name in original_states:
                    param.data.copy_(original_states[name])
                if name in original_requires_grad:
                    param.requires_grad_(original_requires_grad[name])

            if was_training:
                model.train()
            else:
                model.eval()

        return final_predictions

    def _should_optimize(self, param_name: str) -> bool:
        """Check if a parameter should be optimized based on name patterns."""
        return any(pattern in param_name for pattern in self.params_patterns)

    def _compute_tto_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        images_hwc: torch.Tensor,
        brdf_renderer,
    ) -> Optional[torch.Tensor]:
        """Compute the TTO loss (BRDF render loss).

        Args:
            predictions: Model predictions dict.
            images_hwc: Target images [B, S, H, W, 3].
            brdf_renderer: BRDFRenderer instance.

        Returns:
            Scalar loss tensor or None if required inputs are missing.
        """
        required_keys = ["albedo", "normal", "roughness", "metallic", "sg_params"]
        for key in required_keys:
            if key not in predictions or predictions[key] is None:
                return None

        # Get camera/geometry info
        camera_pos = None
        if "pose_enc" in predictions and predictions["pose_enc"] is not None:
            from vggt.heads.brdf_renderer import compute_camera_positions
            camera_pos = compute_camera_positions(predictions["pose_enc"])

        point_map = predictions.get("world_points")

        if camera_pos is None or point_map is None:
            return None

        # Render
        rendered = brdf_renderer(
            albedo=predictions["albedo"],
            normal=predictions["normal"],
            roughness=predictions["roughness"],
            metallic=predictions["metallic"],
            sg_params=predictions["sg_params"],
            point_map=point_map,
            camera_pos=camera_pos,
            shading=predictions.get("shading"),
        )

        # Resize if needed
        if rendered.shape[2:4] != images_hwc.shape[2:4]:
            B, S, H_r, W_r, C = rendered.shape
            H_i, W_i = images_hwc.shape[2], images_hwc.shape[3]
            rendered_flat = rendered.reshape(B * S, H_r, W_r, C).permute(0, 3, 1, 2)
            rendered_flat = F.interpolate(
                rendered_flat, size=(H_i, W_i), mode="bilinear", align_corners=False
            )
            rendered = rendered_flat.permute(0, 2, 3, 1).reshape(B, S, H_i, W_i, C)

        return F.mse_loss(rendered, images_hwc)
