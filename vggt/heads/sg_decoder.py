# SG (Spherical Gaussian) Decoder for VGGT Light Token.
#
# Decodes the light_token embedding [B, 1, 1024] into Spherical Gaussian
# parameters [B, num_lobes, 7] with safe numerical constraints:
#   - Direction (3): L2 normalized to unit sphere
#   - Sharpness (1): Sigmoid mapped to [sharpness_min, sharpness_max]
#   - Amplitude (3): Softplus to ensure non-negative

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGDecoder(nn.Module):
    """Decode light_token into Spherical Gaussian (SG) parameters.

    Each SG lobe is parameterized by 7 values:
        - direction (3D unit vector on the sphere)
        - sharpness (bandwidth / concentration, positive scalar)
        - amplitude / color (3-channel RGB radiance, non-negative)

    The total output dimension per batch = num_lobes * 7.

    Args:
        embed_dim: Dimension of the input light token (default: 1024).
        num_lobes: Number of SG lobes (default: 24).
        hidden_dim: Hidden layer dimension in the MLP (default: 512).
        sharpness_min: Minimum sharpness value (default: 1.0).
        sharpness_max: Maximum sharpness value (default: 1000.0).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_lobes: int = 24,
        hidden_dim: int = 512,
        sharpness_min: float = 1.0,
        sharpness_max: float = 1000.0,
    ):
        super().__init__()
        self.num_lobes = num_lobes
        self.params_per_lobe = 7  # direction(3) + sharpness(1) + amplitude(3)
        out_dim = num_lobes * self.params_per_lobe

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.sharpness_min = sharpness_min
        self.sharpness_max = sharpness_max

        # Initialize output layer with small weights to start near the safe region
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, light_token: torch.Tensor) -> torch.Tensor:
        """Decode light token to SG parameters.

        Args:
            light_token: [B, S, embed_dim]

        Returns:
            sg_params: [B, S, num_lobes, 7]
                - [:, :, 0:3]  direction (unit vector)
                - [:, :, 3:4]  sharpness ∈ [sharpness_min, sharpness_max]
                - [:, :, 4:7]  amplitude (non-negative RGB)
        """
        B, S, C = light_token.shape
        x = light_token.view(-1, C)  # [B*S, embed_dim]

        raw = self.mlp(x)  # [B*S, num_lobes * 7]
        raw = raw.view(B, S, self.num_lobes, self.params_per_lobe)  # [B, S, num_lobes, 7]

        # --- Direction: L2 normalize to unit sphere ---
        direction_raw = raw[..., :3]
        direction = F.normalize(direction_raw, p=2, dim=-1, eps=1e-8)

        # --- Sharpness: Sigmoid -> [min, max] ---
        sharpness_raw = raw[..., 3:4]
        sharpness = self.sharpness_min + (
            self.sharpness_max - self.sharpness_min
        ) * torch.sigmoid(sharpness_raw)

        # --- Amplitude: Softplus to ensure non-negative ---
        amplitude_raw = raw[..., 4:7]
        amplitude = F.softplus(amplitude_raw)

        return torch.cat([direction, sharpness, amplitude], dim=-1)  # [B, S, num_lobes, 7]

    @staticmethod
    def split_sg_params(sg_params: torch.Tensor):
        """Utility to split SG parameter tensor into components.

        Args:
            sg_params: [B, num_lobes, 7] or [..., 7]

        Returns:
            direction: [..., 3] unit vectors
            sharpness: [..., 1] positive scalars
            amplitude: [..., 3] non-negative RGB
        """
        direction = sg_params[..., :3]
        sharpness = sg_params[..., 3:4]
        amplitude = sg_params[..., 4:7]
        return direction, sharpness, amplitude
