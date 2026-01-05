import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = cond_dim

        self.norm = nn.LayerNorm(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, C, H, W]
        cond: [B, cond_dim]
        """
        B, C = x.shape[:2]

        cond = self.norm(cond)
        gamma_beta = self.mlp(cond)   # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)

        return x * (1.0 + gamma) + beta