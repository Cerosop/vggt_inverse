# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
from typing import Optional as _Optional
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
target_indices = {4, 11, 17, 23}


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_lora=False,
        lora_rank=16,
        lora_alpha=32,
        lora_tail_layers=0,
        lora_tail_rank=64,
        lora_global_base_rank=None,
        enable_light_token=False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

        # Light token: learnable global illumination embedding
        # Only participates in Global Attention (no RoPE applied)
        self.enable_light_token = enable_light_token
        if enable_light_token:
            self.light_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            nn.init.normal_(self.light_token, std=1e-6)
        else:
            self.light_token = None

        # Resolved global base rank: use lora_global_base_rank if provided, else lora_rank
        _global_base_rank = lora_global_base_rank if lora_global_base_rank is not None else lora_rank

        # When training the light token, the per-frame pathway is frozen entirely:
        # no LoRA is added to the frame blocks so only the global pathway adapts.
        # (Has no effect unless LoRA is enabled.)
        self.lora_skip_frame = enable_lora and enable_light_token

        # LoRA dual-path support
        self.enable_lora = enable_lora
        if enable_lora:
            from vggt.layers.lora import LoRABlock
            if self.lora_skip_frame:
                # Frame blocks stay frozen (no adapter); the LoRA frame step routes
                # through the original frozen frame_blocks in the forward pass.
                self.lora_frame_blocks = None
            else:
                self.lora_frame_blocks = nn.ModuleList([
                    LoRABlock(
                        blk,
                        rank=lora_tail_rank if (lora_tail_layers > 0 and i >= depth - lora_tail_layers) else lora_rank,
                        alpha=lora_alpha,
                    )
                    for i, blk in enumerate(self.frame_blocks)
                ])
            self.lora_global_blocks = nn.ModuleList([
                LoRABlock(
                    blk,
                    rank=lora_tail_rank if (lora_tail_layers > 0 and i >= depth - lora_tail_layers) else _global_base_rank,
                    alpha=lora_alpha,
                )
                for i, blk in enumerate(self.global_blocks)
            ])
        else:
            self.lora_frame_blocks = None
            self.lora_global_blocks = None

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], _Optional[List[torch.Tensor]], int, _Optional[torch.Tensor]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], list[torch.Tensor] | None, int, Tensor | None):
                - output_list: intermediates from the original (frozen) attention blocks.
                - lora_output_list: intermediates from the LoRA-adapted blocks (None if LoRA disabled).
                - patch_start_idx: where patch tokens begin.
                - light_token_out: updated light token [B, 1, C] or None.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # Decide whether to run the original (frozen) transformer path.
        # During training with LoRA enabled, we ONLY need the LoRA path —
        # the original path's outputs are not used for any loss, so we skip
        # it entirely to save ~50% of the Aggregator compute.
        run_original_path = not (self.training and self.enable_lora)

        # Initialize LoRA tokens as a separate copy if LoRA is enabled
        if self.enable_lora:
            lora_tokens = tokens if not run_original_path else tokens.clone()
        else:
            lora_tokens = None

        # Initialize light token for global attention
        # In LoRA mode it threads through the LoRA path; in full mode it threads through
        # the original path. Either way we maintain a [B*S, 1, C] running representation
        # for frame blocks and reshape to [B, S, C] for global blocks.
        _light_token_out = None
        if self.enable_light_token and self.light_token is not None:
            _light_token_current = self.light_token.expand(B * S, -1, -1)
        else:
            _light_token_current = None

        # Light token is routed through whichever path is actually being executed.
        _lt_through_original = run_original_path and (_light_token_current is not None) and (not self.enable_lora)
        _lt_through_lora = self.enable_lora and (_light_token_current is not None)

        frame_idx = 0
        global_idx = 0
        output_list = [] if run_original_path else None
        lora_output_list = [] if self.enable_lora else None

        for _ in range(self.aa_block_num):
            frame_intermediates = None
            global_intermediates = None
            lora_frame_intermediates = None
            lora_global_intermediates = None

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    if run_original_path:
                        if _lt_through_original:
                            tokens, frame_idx, frame_intermediates, _light_token_current = self._process_frame_attention(
                                tokens, B, S, P, C, frame_idx, pos=pos,
                                light_token=_light_token_current,
                            )
                        else:
                            tokens, frame_idx, frame_intermediates, _ = self._process_frame_attention(
                                tokens, B, S, P, C, frame_idx, pos=pos
                            )
                    else:
                        frame_idx += self.aa_block_size

                    if self.enable_lora:
                        if _lt_through_lora:
                            # light_token enters frame block as [B*S, 1, C]
                            lora_tokens, _, lora_frame_intermediates, _light_token_current = self._process_frame_attention_lora(
                                lora_tokens, B, S, P, C, frame_idx - self.aa_block_size, pos=pos,
                                light_token=_light_token_current,
                            )
                        else:
                            lora_tokens, _, lora_frame_intermediates, _ = self._process_frame_attention_lora(
                                lora_tokens, B, S, P, C, frame_idx - self.aa_block_size, pos=pos
                            )

                elif attn_type == "global":
                    if run_original_path:
                        if _lt_through_original:
                            # [B*S, 1, C] -> [B, S, C] before global attention
                            if _light_token_current.dim() == 3 and _light_token_current.shape[0] == B * S:
                                _light_token_current = _light_token_current.reshape(B, S, C)
                            tokens, global_idx, global_intermediates, _light_token_current = self._process_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos,
                                light_token=_light_token_current,
                            )
                            _light_token_current = _light_token_current.reshape(B * S, 1, C)
                        else:
                            tokens, global_idx, global_intermediates, _ = self._process_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos
                            )
                    else:
                        global_idx += self.aa_block_size

                    if self.enable_lora:
                        if _lt_through_lora:
                            # Convert light token from [B*S, 1, C] -> [B, S, C] for global block
                            if _light_token_current.dim() == 3 and _light_token_current.shape[0] == B * S:
                                _light_token_current = _light_token_current.reshape(B, S, C)

                            lora_tokens, _, lora_global_intermediates, _light_token_current = self._process_global_attention_lora(
                                lora_tokens, B, S, P, C, global_idx - self.aa_block_size, pos=pos,
                                light_token=_light_token_current,
                            )
                            # Convert output back to [B*S, 1, C] for the next frame block, or keep as [B, S, C] for final output
                            _light_token_current = _light_token_current.reshape(B * S, 1, C)
                        else:
                            lora_tokens, _, lora_global_intermediates, _ = self._process_global_attention_lora(
                                lora_tokens, B, S, P, C, global_idx - self.aa_block_size, pos=pos
                            )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if run_original_path and (global_idx - 1) in target_indices:
                for i in range(len(frame_intermediates)):
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)

            if self.enable_lora and (global_idx - 1) in target_indices:
                for i in range(len(lora_frame_intermediates)):
                    concat_inter = torch.cat([lora_frame_intermediates[i], lora_global_intermediates[i]], dim=-1)
                    lora_output_list.append(concat_inter)


        # Final light token output
        if _light_token_current is not None:
            # Reshape final [B*S, 1, C] to [B, S, C]
            _light_token_out = _light_token_current.reshape(B, S, C)
        else:
            _light_token_out = None

        return output_list, lora_output_list, self.patch_start_idx, _light_token_out

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, light_token=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions (reshape, not view: after the light
        # token is split off the tensor can be non-contiguous).
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B * S, P, 2)

        # Append light token (no RoPE position — set to 0)
        use_lt = light_token is not None
        if use_lt:
            tokens = torch.cat([tokens, light_token], dim=1)  # [B*S, P+1, C]
            if pos is not None:
                lt_pos = torch.zeros(B * S, 1, 2, device=pos.device, dtype=pos.dtype)
                pos = torch.cat([pos, lt_pos], dim=1)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            if use_lt:
                patch_tokens_out = tokens[:, :-1, :]
                light_token = tokens[:, -1:, :]
                intermediates.append(patch_tokens_out.reshape(B, S, P, C))
            else:
                intermediates.append(tokens.reshape(B, S, P, C))

        if use_lt:
            tokens = patch_tokens_out

        return tokens, frame_idx, intermediates, light_token

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, light_token=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).

        If light_token is provided ([B, S, C]), it is appended to the token
        sequence before attention (without RoPE) and split back after, mirroring
        the LoRA path. This lets the original-block (full-finetune) path also
        update the light token.
        """
        # reshape (not view): after the light token is split off the tensor can be
        # non-contiguous, which would make .view() raise.
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S * P, 2)

        use_lt = light_token is not None
        if use_lt:
            tokens = torch.cat([tokens, light_token], dim=1)  # [B, S*P + S, C]
            if pos is not None:
                lt_pos = torch.zeros(B, S, 2, device=pos.device, dtype=pos.dtype)
                pos = torch.cat([pos, lt_pos], dim=1)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            if use_lt:
                patch_tokens_out = tokens[:, :-S, :]
                light_token = tokens[:, -S:, :]
                intermediates.append(patch_tokens_out.reshape(B, S, P, C))
            else:
                intermediates.append(tokens.reshape(B, S, P, C))

        if use_lt:
            tokens = patch_tokens_out

        return tokens, global_idx, intermediates, light_token

    # ------------------------------------------------------------------
    # LoRA-adapted attention processing (mirrors original but uses LoRA)
    # ------------------------------------------------------------------

    def _process_frame_attention_lora(self, tokens, B, S, P, C, frame_idx, pos=None, light_token=None):
        """Process frame attention using LoRA-adapted blocks.

        When ``lora_skip_frame`` is set (light-token training), no LoRA is applied
        to the frame blocks: the original frozen ``frame_blocks`` are used instead,
        so the per-frame representation stays fixed and only the global pathway adapts.
        """
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B * S, P, 2)

        # Append light token (no RoPE position — set to 0)
        use_lt = light_token is not None
        if use_lt:
            tokens = torch.cat([tokens, light_token], dim=1)  # [B*S, P+1, C]
            if pos is not None:
                lt_pos = torch.zeros(B * S, 1, 2, device=pos.device, dtype=pos.dtype)
                pos = torch.cat([pos, lt_pos], dim=1)

        # Frozen frame blocks when frame LoRA is skipped, else the LoRA-adapted ones.
        frame_block_list = self.frame_blocks if self.lora_skip_frame else self.lora_frame_blocks

        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(frame_block_list[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = frame_block_list[frame_idx](tokens, pos=pos)
            frame_idx += 1
            
            if use_lt:
                patch_tokens_out = tokens[:, :-1, :]
                light_token = tokens[:, -1:, :]
                intermediates.append(patch_tokens_out.reshape(B, S, P, C))
            else:
                intermediates.append(tokens.reshape(B, S, P, C))

        if use_lt:
            tokens = patch_tokens_out
            return tokens, frame_idx, intermediates, light_token
            
        return tokens, frame_idx, intermediates, light_token

    def _process_global_attention_lora(self, tokens, B, S, P, C, global_idx, pos=None, light_token=None):
        """Process global attention using LoRA-adapted blocks.

        If light_token is provided, it is appended to the token sequence
        before attention (without RoPE) and split back after attention.
        This allows the light token to attend to all patch tokens across views.
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S * P, 2)

        # Append light token (no RoPE position — set to 0)
        use_lt = light_token is not None
        if use_lt:
            tokens = torch.cat([tokens, light_token], dim=1)  # [B, S*P + S, C]
            if pos is not None:
                lt_pos = torch.zeros(B, S, 2, device=pos.device, dtype=pos.dtype)
                pos = torch.cat([pos, lt_pos], dim=1)  # [B, S*P + S, 2]

        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.lora_global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.lora_global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1

            if use_lt:
                # Split out light token and patch tokens
                patch_tokens_out = tokens[:, :-S, :]   # [B, S*P, C]
                light_token = tokens[:, -S:, :]        # [B, S, C]
                intermediates.append(patch_tokens_out.reshape(B, S, P, C))
            else:
                intermediates.append(tokens.reshape(B, S, P, C))

        if use_lt:
            # Keep only patch tokens. Light tokens are passed separately.
            tokens = patch_tokens_out

        if tokens.shape[1] != S * P:
            raise RuntimeError(
                f"Global LoRA output token length mismatch: got {tokens.shape[1]}, expected {S * P} "
                f"(B={B}, S={S}, P={P}, use_lt={use_lt})."
            )

        return tokens, global_idx, intermediates, light_token


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
