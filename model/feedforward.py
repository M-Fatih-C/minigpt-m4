import torch
import torch.nn as nn

from config import (
    DROPOUT,
    N_EMBD,
)


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        # Transformer'da klasik olarak FFN hidden
        # dimension embedding'in ~4 katıdır.
        hidden_size = 4 * N_EMBD

        self.network = nn.Sequential(

            nn.Linear(
                N_EMBD,
                hidden_size,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_size,
                N_EMBD,
            ),

            nn.Dropout(
                DROPOUT
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)