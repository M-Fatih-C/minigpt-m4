import torch
import torch.nn as nn

from config import (
    BLOCK_SIZE,
    N_EMBD,
    VOCAB_SIZE,
)


class GPTEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        # Her token ID için öğrenilebilir bir vektör.
        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            N_EMBD,
        )

        # Sequence içindeki her pozisyon için
        # öğrenilebilir başka bir vektör.
        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        # token_ids:
        #
        # [B, T]
        #
        # örnek:
        # [32, 256]

        batch_size, sequence_length = (
            token_ids.shape
        )

        if sequence_length > BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds BLOCK_SIZE={BLOCK_SIZE}"
            )

        # [B, T]
        #      ↓
        # [B, T, C]
        token_embeddings = (
            self.token_embedding(token_ids)
        )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        # [T]
        #  ↓
        # [T, C]
        position_embeddings = (
            self.position_embedding(positions)
        )

        # Broadcasting:
        #
        # [B, T, C]
        # +
        # [   T, C]
        #
        # =
        #
        # [B, T, C]

        x = (
            token_embeddings
            + position_embeddings
        )

        return x