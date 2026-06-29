# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_inverse=False, enable_lora=False, lora_rank=16, lora_alpha=32,
                 lora_alpha_mult=2.0,
                 lora_tail_layers=0, lora_tail_rank=64,
                 lora_global_base_rank=None,
                 inverse_frames_chunk_size=8, skip_vggt_heads_in_train=False,
                 inverse_head_type=None, inverse_head_pos_embed=None,
                 enable_light_token=False, sg_num_lobes=24, sg_hidden_dim=512,
                 enable_brdf_render=False, brdf_geometry_source="pred",
                 resnext_pretrained=True,
                 resnext_disable_layer34=False,
                 enable_dynamic_weighting=False,
                 material_decoder="dpt", lighting_mode="sg",
                 d4rt_decoder_dim=1024, d4rt_num_layers=8, d4rt_num_heads=16,
                 light_env_h=8, light_env_w=16, light_spatial_h=60, light_spatial_w=80,
                 num_light_samples=2048, num_render_pixels=64,
                 num_material_samples=0,
                 enable_per_pixel_render=False,
                 per_pixel_render_specular=False,
                 d4rt_cross_frame=False,
                 d4rt_render_demo_upsample=4,
                 d4rt_grad_checkpoint=False,
                 tto_config=None):
        super().__init__()

        self.skip_vggt_heads_in_train = skip_vggt_heads_in_train
        self.enable_brdf_render = enable_brdf_render
        # d4rt per-pixel render: include the SPECULAR term (needs geometry for the
        # view direction). Independent of enable_brdf_render (the old SG-render flag).
        self.per_pixel_render_specular = per_pixel_render_specular
        self.brdf_geometry_source = brdf_geometry_source
        self.enable_light_token = enable_light_token
        self.material_decoder = material_decoder   # "dpt" (old) | "d4rt" (new)
        self.lighting_mode = lighting_mode         # "sg" (old) | "per_pixel_env" (new)

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            enable_lora=enable_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_alpha_mult=lora_alpha_mult,
            lora_tail_layers=lora_tail_layers, lora_tail_rank=lora_tail_rank,
            lora_global_base_rank=lora_global_base_rank,
            enable_light_token=enable_light_token,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # Inverse rendering heads (NEW)
        # Built only when at least one branch still uses the old path
        # (DPT materials or SG lighting). In pure-d4rt mode both branches are
        # handled by d4rt_heads, so this (incl. its ResNeXt encoder) is skipped.
        self.inverse_heads = None
        if enable_inverse and (material_decoder == "dpt" or lighting_mode == "sg"):
            from vggt.heads.inverse_heads import InverseHeads
            self.inverse_heads = InverseHeads(
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                frames_chunk_size=inverse_frames_chunk_size,
                head_type_config=inverse_head_type,
                head_pos_embed_config=inverse_head_pos_embed,
                enable_sg=enable_light_token,
                sg_num_lobes=sg_num_lobes,
                sg_hidden_dim=sg_hidden_dim,
                sg_embed_dim=embed_dim,
                resnext_pretrained=resnext_pretrained,
                resnext_disable_layer34=resnext_disable_layer34,
                enable_dynamic_weighting=enable_dynamic_weighting,
            )

        # d4rt cross-attention inverse heads (NEW, switchable). Built only when a
        # branch is set to "d4rt"/"per_pixel_env"; the old path above is untouched.
        self.d4rt_heads = None
        if enable_inverse and (material_decoder == "d4rt" or lighting_mode == "per_pixel_env"):
            from vggt.heads.d4rt_inverse_heads import D4RTInverseHeads
            self.d4rt_heads = D4RTInverseHeads(
                dim_in=2 * embed_dim,
                decoder_dim=d4rt_decoder_dim,
                num_layers=d4rt_num_layers,
                num_heads=d4rt_num_heads,
                patch_size=patch_size,
                enable_material=(material_decoder == "d4rt"),
                enable_lighting=(lighting_mode == "per_pixel_env"),
                env_h=light_env_h, env_w=light_env_w,
                light_spatial_h=light_spatial_h, light_spatial_w=light_spatial_w,
                num_light_samples=num_light_samples,
                num_material_samples=num_material_samples,
                # Render-input (env tile) computation is gated by its own flag, NOT
                # enable_brdf_render. enable_brdf_render only adds the geometry (point/
                # camera heads) needed for the SPECULAR term; the diffuse render is
                # geometry-free, so diffuse-only render keeps the d4rt memory low.
                enable_render=(lighting_mode == "per_pixel_env" and enable_per_pixel_render),
                num_render_pixels=num_render_pixels,
                render_demo_upsample=d4rt_render_demo_upsample,
                enable_dynamic_weighting=enable_dynamic_weighting,
                cross_frame=d4rt_cross_frame,
                grad_checkpoint=d4rt_grad_checkpoint,
            )

        # TTO config (used at inference only)
        self.tto_config = tto_config
        self._tto_optimizer = None

    def _cfg_get(self, cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _is_tto_enabled(self) -> bool:
        return bool(self._cfg_get(self.tto_config, "enable", self._cfg_get(self.tto_config, "enabled", False)))

    def _get_tto_optimizer(self):
        if self._tto_optimizer is None:
            from vggt.utils.tto import TestTimeOptimizer

            iterations = int(self._cfg_get(self.tto_config, "iterations", 50))
            lr = float(self._cfg_get(self.tto_config, "lr", 7e-4))
            params_patterns = self._cfg_get(self.tto_config, "params_patterns", None)
            self._tto_optimizer = TestTimeOptimizer(
                iterations=iterations,
                lr=lr,
                params_patterns=params_patterns,
            )
        return self._tto_optimizer

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, _disable_tto: bool = False):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
                - sg_params (torch.Tensor): SG lighting parameters [B, num_lobes, 7] (if light_token enabled)

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if (not self.training) and (not _disable_tto) and self._is_tto_enabled():
            tto_optimizer = self._get_tto_optimizer()
            return tto_optimizer.optimize(self, images, query_points=query_points)

        aggregated_tokens_list, lora_tokens_list, patch_start_idx, light_token_out = self.aggregator(images)

        predictions = {}

        # --- Original VGGT heads (frozen, skip during inverse-only training) ---
        # aggregated_tokens_list is None when training with LoRA (original path skipped)
        run_vggt_heads = (
            not (self.training and self.skip_vggt_heads_in_train)
            and aggregated_tokens_list is not None
        )

        if run_vggt_heads:
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]
                    predictions["pose_enc_list"] = pose_enc_list

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["world_points"] = pts3d
                    predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf

        # --- Geometry for BRDF render loss ---
        # When using predicted geometry ("pred"), run the (frozen) camera/point heads
        # under no_grad so the renderer gets fixed geometry without the render loss
        # back-propagating into the backbone. Works in both full mode (original tokens)
        # and LoRA mode (LoRA tokens) — VGGT heads are otherwise skipped during training.
        # In "gt" mode the dataset supplies geometry, so this is skipped.
        # Triggered by the old SG render (enable_brdf_render) OR the d4rt per-pixel
        # render's specular term (per_pixel_render_specular) — either needs geometry.
        if (self.training
                and (self.enable_brdf_render or getattr(self, "per_pixel_render_specular", False))
                and getattr(self, "brdf_geometry_source", "pred") == "pred"):
            geom_tokens = lora_tokens_list if lora_tokens_list is not None else aggregated_tokens_list
            if geom_tokens is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        if self.camera_head is not None and "pose_enc" not in predictions:
                            pose_enc_list = self.camera_head(geom_tokens)
                            predictions["pose_enc"] = pose_enc_list[-1]

                        if self.point_head is not None and "world_points" not in predictions:
                            pts3d, pts3d_conf = self.point_head(
                                geom_tokens, images=images, patch_start_idx=patch_start_idx
                            )
                            predictions["world_points"] = pts3d

        # --- Inverse rendering heads ---
        tokens_for_inverse = lora_tokens_list if lora_tokens_list is not None else aggregated_tokens_list
        # Old DPT/SG path: run when materials use "dpt" OR lighting uses "sg".
        if self.inverse_heads is not None and (self.material_decoder == "dpt" or self.lighting_mode == "sg"):
            inverse_preds = self.inverse_heads(
                tokens_for_inverse, images, patch_start_idx,
                light_token=light_token_out,
            )
            if self.material_decoder == "dpt":
                predictions.update(inverse_preds)        # materials (+ sg if any)
            elif "sg_params" in inverse_preds:
                predictions["sg_params"] = inverse_preds["sg_params"]  # keep only sg
        # New d4rt path: material (d4rt) and/or lighting (per_pixel_env).
        if self.d4rt_heads is not None:
            predictions.update(self.d4rt_heads(tokens_for_inverse, images, patch_start_idx))

        if not self.training:
            predictions["images"] = images

        return predictions
