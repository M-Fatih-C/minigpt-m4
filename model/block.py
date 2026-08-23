import torch
import torch.nn as nn

from config import N_EMBD

from model.attention import (
    MultiHeadAttention,
)

from model.feedforward import (
    FeedForward,
)


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            N_EMBD
        )

        self.attention = (
            MultiHeadAttention()
        )

        self.layer_norm_2 = nn.LayerNorm(
            N_EMBD
        )

        self.feed_forward = (
            FeedForward()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Pre-LayerNorm Transformer
        #
        # x --------+
        #           |
        # LayerNorm |
        #     ↓     |
        # Attention |
        #     ↓     |
        #     + <---+
        #
        x = (
            x
            + self.attention(
                self.layer_norm_1(x)
            )
        )

        # x --------+
        #           |
        # LayerNorm |
        #     ↓     |
        #    FFN    |
        #     ↓     |
        #     + <---+
        #
        x = (
            x
            + self.feed_forward(
                self.layer_norm_2(x)
            )
        )

        return x