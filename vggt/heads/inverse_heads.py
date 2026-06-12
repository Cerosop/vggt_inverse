# Inverse Rendering Heads for VGGT.
#
# Ported from MVInverse with modifications:
# - intermediate_layer_idx changed to [0, 1, 2, 3] (VGGT has 4 intermediates after filtering)
# - Added chunked inference (from VGGT DPTHead)
# - Configurable head type: DPTHead or DPTHeadRes per head (via head_type_config)
# - Configurable positional embedding per head (via head_pos_embed_config)
# - No confidence output (MVInverse style)
# - Includes ResNeXt-101 encoder for DPTHeadRes heads

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union, Optional
from torchvision.models import resnext101_32x8d


# ============================================================================
# InverseHeads: Container for all 5 inverse rendering heads + ResNeXt encoder
# ============================================================================

class InverseHeads(nn.Module):
    """Container module for MVInverse-style inverse rendering heads.

    Includes:
    - ResNeXt-101 (32×8d) encoder for multi-scale CNN features (only if needed)
    - 5 DPT-based prediction heads: albedo, metallic, roughness, normal, shading
    - Chunked inference support for memory-efficient processing
    - Configurable head type (dpt / dpt_res) per head
    - Configurable positional embedding per head

    Args:
        dim_in: Input token dimension (2 * embed_dim = 2048 for ViT-L).
        patch_size: Patch size from the ViT encoder.
        frames_chunk_size: Number of frames to process at once in DPT heads.
        head_type_config: Dict mapping head name -> "dpt" or "dpt_res".
        head_pos_embed_config: Dict mapping head name -> bool (enable pos_embed).
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        frames_chunk_size: int = 8,
        head_type_config: Optional[Dict[str, str]] = None,
        head_pos_embed_config: Optional[Dict[str, bool]] = None,
        enable_sg: bool = False,
        sg_num_lobes: int = 24,
        sg_hidden_dim: int = 512,
        sg_embed_dim: int = 1024,
        resnext_pretrained: bool = True,
        resnext_disable_layer34: bool = False,
        enable_dynamic_weighting: bool = False,
    ):
        super().__init__()
        self.frames_chunk_size = frames_chunk_size

        # Default head type config
        default_head_types = {
            "albedo": "dpt_res",
            "metallic": "dpt",
            "roughness": "dpt",
            "normal": "dpt",
            "shading": "dpt_res",
        }
        head_types = default_head_types
        if head_type_config is not None:
            head_types.update(head_type_config)

        # Default pos_embed config (all False by default)
        default_pos_embed = {
            "albedo": False,
            "metallic": False,
            "roughness": False,
            "normal": False,
            "shading": False,
        }
        pos_embed_cfg = default_pos_embed
        if head_pos_embed_config is not None:
            pos_embed_cfg.update(head_pos_embed_config)

        # Determine which heads use ResNeXt
        self.res_head_names = {name for name, htype in head_types.items() if htype == "dpt_res"}

        # Only build ResNeXt encoder if at least one head needs it.
        # When `resnext_pretrained=True`, init from WSL/ImageNet pretrained weights.
        # A downstream checkpoint that contains res_encoder.* keys will overwrite
        # this initial state via `load_state_dict(..., strict=False)`.
        if self.res_head_names:
            self.res_encoder = _make_pretrained_resnext101_wsl(
                use_pretrained=resnext_pretrained,
            )
        else:
            self.res_encoder = None

        # Head output configs
        head_configs = {
            "albedo": {"output_dim": 3, "activation": "sigmoid"},
            "metallic": {"output_dim": 1, "activation": "sigmoid"},
            "roughness": {"output_dim": 1, "activation": "sigmoid"},
            "normal": {"output_dim": 3, "activation": "tanh"},
            "shading": {"output_dim": 3, "activation": "sigmoid"},
        }

        # Build heads
        for head_name, cfg in head_configs.items():
            htype = head_types[head_name]
            use_pos = pos_embed_cfg.get(head_name, False)

            if htype == "dpt_res":
                head = InverseDPTHeadRes(
                    dim_in=dim_in, patch_size=patch_size,
                    output_dim=cfg["output_dim"], activation=cfg["activation"],
                    pos_embed=use_pos,
                    disable_layer34=resnext_disable_layer34,
                )
            else:
                head = InverseDPTHead(
                    dim_in=dim_in, patch_size=patch_size,
                    output_dim=cfg["output_dim"], activation=cfg["activation"],
                    pos_embed=use_pos,
                )
            setattr(self, f"{head_name}_head", head)

        # Dynamic loss weighting (Kendall et al. 2018 uncertainty weighting).
        # One learnable log-variance parameter per head; the loss module reads
        # this from the predictions dict and applies:
        #     L_total = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
        # where s_i = log(σ_i²). Tasks with consistently large loss naturally
        # learn larger σ → smaller weight, so they don't drown out the others.
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.dyn_task_names = ["albedo", "metallic", "roughness", "normal", "shading"]
        if enable_dynamic_weighting:
            self.task_log_var = nn.Parameter(torch.zeros(len(self.dyn_task_names)))
        else:
            self.register_parameter("task_log_var", None)

        # Normalization buffers for ResNeXt input
        self.register_buffer(
            "_resnet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        # All head names for iteration
        self.head_names = ["albedo", "metallic", "roughness", "normal", "shading"]

        # SG Decoder (for light token -> SG parameters)
        self.sg_decoder = None
        if enable_sg:
            from vggt.heads.sg_decoder import SGDecoder
            self.sg_decoder = SGDecoder(
                embed_dim=sg_embed_dim,
                num_lobes=sg_num_lobes,
                hidden_dim=sg_hidden_dim,
            )

    def _get_resnext_features(self, images_norm: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from ResNeXt-101.

        Args:
            images_norm: Normalized images [B*S, 3, H, W].

        Returns:
            List of 4 feature tensors at different scales.
        """
        H, W = images_norm.shape[-2:]
        new_H, new_W = H // 7 * 8, W // 7 * 8
        x_resized = F.interpolate(images_norm, (new_H, new_W), mode='bilinear', align_corners=False)

        layer_1 = self.res_encoder.layer1(x_resized)
        layer_2 = self.res_encoder.layer2(layer_1)
        layer_3 = self.res_encoder.layer3(layer_2)
        layer_4 = self.res_encoder.layer4(layer_3)
        return [layer_1, layer_2, layer_3, layer_4]

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        heads_to_predict: Optional[List[str]] = None,
        light_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through inverse rendering heads.

        Args:
            aggregated_tokens_list: List of intermediate token tensors [B, S, P, 2C].
            images: Input images [B, S, 3, H, W] in [0, 1] range.
            patch_start_idx: Starting index for patch tokens.
            heads_to_predict: Optional list of head names to predict. If None, all heads run.
            light_token: Optional light token [B, 1, C] from aggregator for SG decoding.

        Returns:
            Dict mapping head names to predictions.
        """
        B, S, _, H, W = images.shape
        active_heads = heads_to_predict if heads_to_predict else self.head_names

        predictions = {}

        # Only compute ResNeXt features if needed
        res_features = None
        if self.res_encoder is not None and any(h in self.res_head_names for h in active_heads):
            imgs_norm = (images - self._resnet_mean) / self._resnet_std
            imgs_flat = imgs_norm.reshape(B * S, 3, H, W)
            res_features = self._get_resnext_features(imgs_flat)

        for head_name in active_heads:
            head = getattr(self, f"{head_name}_head")

            if head_name in self.res_head_names:
                pred = head(
                    aggregated_tokens_list, images,
                    res_features=res_features,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.frames_chunk_size,
                )
            else:
                pred = head(
                    aggregated_tokens_list, images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.frames_chunk_size,
                )

            # Post-process normal to unit vector
            if head_name == "normal":
                pred = F.normalize(pred, p=2, dim=-1, eps=1e-8)

            predictions[head_name] = pred

        # Decode SG parameters from light token
        if self.sg_decoder is not None and light_token is not None:
            predictions["sg_params"] = self.sg_decoder(light_token)

        # Expose dynamic-weighting parameter to the loss module (if enabled).
        # DDP-safety: we must touch `task_log_var` inside this forward so DDP's
        # `find_unused_parameters` traversal sees it as USED in forward and does
        # not pre-mark it ready (which would later collide with the grad coming
        # back from the loss module, raising "marked as ready twice").
        # The ghost is exactly zero, so it has no numerical effect.
        if self.enable_dynamic_weighting and self.task_log_var is not None:
            ghost = (self.task_log_var.sum() * 0.0)
            for _name in self.dyn_task_names:
                if _name in predictions:
                    predictions[_name] = predictions[_name] + ghost
                    break
            predictions["task_log_var"] = self.task_log_var
            predictions["task_log_var_names"] = self.dyn_task_names

        return predictions


# ============================================================================
# Positional Embedding utilities (ported from VGGT DPTHead)
# ============================================================================

def _create_uv_grid(width, height, aspect_ratio=None, dtype=None, device=None):
    """Create a normalized UV grid of shape (height, width, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)
    return uv_grid


def _position_grid_to_embed(pos_grid, embed_dim, omega_0=100):
    """Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)."""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


def _make_sincos_pos_embed(embed_dim, pos, omega_0=100):
    """Generate 1D positional embedding from positions using sine and cosine functions."""
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos.double(), omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()


def _apply_pos_embed(x, W, H, ratio=0.1):
    """Apply UV sincos positional embedding to feature map x.

    Args:
        x: [N, C, h, w] feature map
        W: Original image width (for aspect ratio)
        H: Original image height (for aspect ratio)
        ratio: Scaling factor for the positional embedding

    Returns:
        x + pos_embed: [N, C, h, w]
    """
    patch_w = x.shape[-1]
    patch_h = x.shape[-2]
    pos_embed = _create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pos_embed = _position_grid_to_embed(pos_embed, x.shape[1])  # [h, w, C]
    pos_embed = pos_embed * ratio
    pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)  # [N, C, h, w]
    return x + pos_embed


# ============================================================================
# InverseDPTHead: DPT head for metallic / roughness / normal
# ============================================================================

class InverseDPTHead(nn.Module):
    """DPT Head for inverse rendering (no ResNeXt, no confidence).

    Adapted from MVInverse DPTHead with:
    - intermediate_layer_idx = [0, 1, 2, 3] for VGGT's filtered intermediates
    - Chunked inference support
    - sigmoid / tanh activation only
    - Optional positional embedding (VGGT-style UV sincos)
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "sigmoid",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [0, 1, 2, 3],
        pos_embed: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.intermediate_layer_idx = intermediate_layer_idx
        self.pos_embed = pos_embed

        self.norm = nn.LayerNorm(dim_in)

        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, kernel_size=1),
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        all_preds = []
        for start in range(0, S, frames_chunk_size):
            end = min(start + frames_chunk_size, S)
            chunk_pred = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, start, end
            )
            all_preds.append(chunk_pred)

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start: int = None,
        frames_end: int = None,
    ) -> torch.Tensor:
        if frames_start is not None and frames_end is not None:
            images = images[:, frames_start:frames_end].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start is not None and frames_end is not None:
                x = x[:, frames_start:frames_end]
            x = x.reshape(B * S, x.shape[2], x.shape[3])
            x = self.norm(x)
            # P1: soft saturation on transformer intermediate to suppress
            #     OOD outlier channels carried over from VGGT geometry pretrain.
            x = torch.tanh(x / 5.0) * 5.0
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            # Apply positional embedding after projects, before resize
            if self.pos_embed:
                x = _apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        out = self._scratch_forward(out, W, H)
        out = _custom_interpolate(
            out,
            (patch_h * self.patch_size, patch_w * self.patch_size),
            mode="bilinear", align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        # P3: clamp logits to keep them out of the sigmoid saturation trap
        #     where gradients vanish and collapse becomes irreversible.
        if self.activation == "sigmoid":
            out = out.clamp(-15.0, 15.0)
            preds = torch.sigmoid(out)
        elif self.activation == "tanh":
            out = out.clamp(-5.0, 5.0)
            preds = torch.tanh(out)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        preds = preds.permute(0, 2, 3, 1)  # (BS, H, W, C)
        preds = preds.view(B, S, *preds.shape[1:])
        return preds

    def _scratch_forward(self, features: List[torch.Tensor], W: int = 0, H: int = 0) -> torch.Tensor:
        l1, l2, l3, l4 = features
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        out = self.scratch.output_conv1(out)
        # Apply positional embedding after output_conv1
        if self.pos_embed and W > 0 and H > 0:
            out = _apply_pos_embed(out, W, H)
        return out


# ============================================================================
# InverseDPTHeadRes: DPT head with ResNeXt fusion for albedo / shading
# ============================================================================

class InverseDPTHeadRes(nn.Module):
    """DPT Head with ResNeXt feature fusion for albedo and shading.

    Same as InverseDPTHead but adds element-wise addition of ResNeXt features
    after the resize step.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        output_dim: int = 3,
        activation: str = "sigmoid",
        features: int = 256,
        in_channels: List[int] = [256, 512, 1024, 2048],
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [0, 1, 2, 3],
        pos_embed: bool = False,
        disable_layer34: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.intermediate_layer_idx = intermediate_layer_idx
        self.pos_embed = pos_embed
        # When True, skip ResNeXt fusion for stages 3 and 4 (i.e. fuse indices
        # 2,3). The deep ResNeXt-101 layer3 has 23 bottleneck blocks whose
        # accumulated BatchNorm/conv stages produce high-norm OOD outliers on
        # uniform regions; disabling fusion at those stages keeps only the
        # cleaner low-level (layer1/2) detail contribution.
        self.disable_layer34 = disable_layer34

        self.norm = nn.LayerNorm(dim_in)

        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels
        ])

        # P2: ResNeXt fusion = Conv(1×1, bias=False) + GroupNorm, gated by a
        # per-channel learnable scalar (zero-init, wrapped in tanh).
        # GroupNorm caps the magnitude of fused features (unlike BN, it does
        # not depend on batch statistics and is bf16-stable).
        # The zero-init gate keeps the ResNeXt path OFF at the start of
        # training so the model first converges using only the DPT path.
        self.fuse_projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1, bias=False),
                nn.GroupNorm(
                    num_groups=min(32, max(1, out_channels[i] // 8)),
                    num_channels=out_channels[i],
                ),
            )
            for i in range(len(in_channels))
        ])
        self.fuse_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1, out_channels[i], 1, 1))
            for i in range(len(in_channels))
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, kernel_size=1),
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        res_features: List[torch.Tensor],
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(
                aggregated_tokens_list, images, res_features, patch_start_idx
            )

        all_preds = []
        for start in range(0, S, frames_chunk_size):
            end = min(start + frames_chunk_size, S)
            chunk_pred = self._forward_impl(
                aggregated_tokens_list, images, res_features,
                patch_start_idx, start, end
            )
            all_preds.append(chunk_pred)

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        res_features: List[torch.Tensor],
        patch_start_idx: int,
        frames_start: int = None,
        frames_end: int = None,
    ) -> torch.Tensor:
        if frames_start is not None and frames_end is not None:
            images = images[:, frames_start:frames_end].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start is not None and frames_end is not None:
                x = x[:, frames_start:frames_end]
            x = x.reshape(B * S, x.shape[2], x.shape[3])
            x = self.norm(x)
            # P1: soft saturation on transformer intermediate.
            x = torch.tanh(x / 5.0) * 5.0
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            # Apply positional embedding after projects, before resize
            if self.pos_embed:
                x = _apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        # P2: gated fusion with ResNeXt features. Replaces the unbounded
        # element-wise add with `out += tanh(gate) * Norm(Conv(rf))`.
        # When chunking, res_features covers all frames, so we need to chunk them too
        for i in range(len(out)):
            # Optionally skip deep stages (layer3 / layer4 fusion) to avoid the
            # OOD-outlier contamination from ResNeXt-101's deep BN stack.
            if self.disable_layer34 and i >= 2:
                continue
            rf = res_features[i]
            if frames_start is not None and frames_end is not None:
                # rf is [B*S_total, C, h, w], we need [B*chunk_S, C, h, w]
                total_bs = rf.shape[0]
                # safely handle dummy tensors where B could be 0
                s_total = total_bs // B if B > 0 else 0
                rf = rf.view(B, s_total, rf.shape[1], rf.shape[2], rf.shape[3])
                rf = rf[:, frames_start:frames_end].contiguous()
                rf = rf.view(B * S, rf.shape[2], rf.shape[3], rf.shape[4])
            fused = self.fuse_projects[i](rf)
            out[i] = out[i] + torch.tanh(self.fuse_gates[i]) * fused

        out = self._scratch_forward(out, W, H)
        out = _custom_interpolate(
            out,
            (patch_h * self.patch_size, patch_w * self.patch_size),
            mode="bilinear", align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        # P3: clamp logits to keep them out of the saturation trap.
        if self.activation == "sigmoid":
            out = out.clamp(-15.0, 15.0)
            preds = torch.sigmoid(out)
        elif self.activation == "tanh":
            out = out.clamp(-5.0, 5.0)
            preds = torch.tanh(out)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        preds = preds.permute(0, 2, 3, 1)
        preds = preds.view(B, S, *preds.shape[1:])
        return preds

    def _scratch_forward(self, features: List[torch.Tensor], W: int = 0, H: int = 0) -> torch.Tensor:
        l1, l2, l3, l4 = features
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        out = self.scratch.output_conv1(out)
        # Apply positional embedding after output_conv1
        if self.pos_embed and W > 0 and H > 0:
            out = _apply_pos_embed(out, W, H)
        return out


# ============================================================================
# Shared utility modules (ported from MVInverse)
# ============================================================================

def _make_pretrained_resnext101_wsl(use_pretrained: bool = True, in_chan: int = 3):
    """Build ResNeXt-101 (32×8d) backbone.

    When ``use_pretrained=True`` (default), tries the following weight sources in order:
      1. Facebook's WSL (Weakly-Supervised Learning on Instagram) checkpoint —
         best transfer-learning init, needs ``torch.hub`` network access on first call.
      2. torchvision's ImageNet1k pretrained weights — local cache, no external repo.
      3. Random init — final fallback if both above fail.

    When a downstream checkpoint already contains ``inverse_heads.res_encoder.*``
    keys (e.g. resuming from ``logs_new/checkpoint.pt``), those weights will
    overwrite this initial state via ``model.load_state_dict(..., strict=False)``.
    Pretrained init therefore only matters when starting from a checkpoint that
    has no ResNeXt entries — e.g. ``weight/vggt_1b_pretrained.pt`` cold start.
    """
    resnet = None
    if use_pretrained:
        # 1) WSL (Instagram billion-image weakly-supervised) — best for transfer.
        try:
            resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
        except Exception as e:
            import logging
            logging.warning(
                f"ResNeXt-101 WSL hub load failed ({e}); "
                "falling back to torchvision ImageNet1k pretrained."
            )
        # 2) torchvision ImageNet1k pretrained.
        if resnet is None:
            try:
                from torchvision.models import ResNeXt101_32X8D_Weights
                resnet = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
            except Exception as e:
                import logging
                logging.warning(
                    f"ResNeXt-101 torchvision ImageNet load failed ({e}); "
                    "falling back to random init."
                )
        # 3) Random init.
        if resnet is None:
            resnet = resnext101_32x8d(weights=None)
    else:
        resnet = resnext101_32x8d(weights=None)

    if in_chan != 3:
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_chan, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )

    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained


def _make_fusion_block(features, size=None, has_residual=True, groups=1):
    return FeatureFusionBlock(
        features, nn.ReLU(inplace=True),
        deconv=False, bn=False, expand=False,
        align_corners=True, size=size,
        has_residual=has_residual, groups=groups,
    )


def _make_scratch(in_shape, out_shape, groups=1):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, 3, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, 3, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, 3, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, 3, padding=1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, groups=groups)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False,
                 align_corners=True, size=None, has_residual=True, groups=1):
        super().__init__()
        self.align_corners = align_corners
        self.has_residual = has_residual
        self.size = size

        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, groups=groups)

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)

        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = _custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def _custom_interpolate(x, size=None, scale_factor=None, mode="bilinear", align_corners=True):
    """Custom interpolate that handles large tensors to avoid INT_MAX issues."""
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        parts = [F.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks]
        return torch.cat(parts, dim=0).contiguous()
    else:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
