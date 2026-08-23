import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v3 import (
    FFN_HIDDEN_SIZE,
    N_EMBD,
)


class SwiGLU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gate_projection = nn.Linear(
            N_EMBD,
            FFN_HIDDEN_SIZE,
            bias=False,
        )

        self.up_projection = nn.Linear(
            N_EMBD,
            FFN_HIDDEN_SIZE,
            bias=False,
        )

        self.down_projection = nn.Linear(
            FFN_HIDDEN_SIZE,
            N_EMBD,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate = F.silu(
            self.gate_projection(x)
        )

        up = self.up_projection(
            x
        )

        return self.down_projection(
            gate * up
        )
