import torch
import torch.nn as nn

from config_v3 import N_EMBD

from model_v3.attention import (
    CausalSelfAttention,
)

from model_v3.feedforward import (
    SwiGLU,
)

from model_v3.norm import (
    RMSNorm,
)


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention_norm = RMSNorm(
            N_EMBD
        )

        self.attention = (
            CausalSelfAttention()
        )

        self.ffn_norm = RMSNorm(
            N_EMBD
        )

        self.feed_forward = SwiGLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = (
            x
            + self.attention(
                self.attention_norm(
                    x
                )
            )
        )

        x = (
            x
            + self.feed_forward(
                self.ffn_norm(
                    x
                )
            )
        )

        return x
