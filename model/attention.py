import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BLOCK_SIZE,
    DROPOUT,
    HEAD_SIZE,
    N_EMBD,
    N_HEAD,
)


class CausalSelfAttentionHead(nn.Module):

    def __init__(
        self,
        head_size: int = HEAD_SIZE,
    ):
        super().__init__()

        self.head_size = head_size

        # X → K
        self.key = nn.Linear(
            N_EMBD,
            head_size,
            bias=False,
        )

        # X → Q
        self.query = nn.Linear(
            N_EMBD,
            head_size,
            bias=False,
        )

        # X → V
        self.value = nn.Linear(
            N_EMBD,
            head_size,
            bias=False,
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        # Causal mask:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        mask = torch.tril(
            torch.ones(
                BLOCK_SIZE,
                BLOCK_SIZE,
                dtype=torch.bool,
            )
        )

        # register_buffer:
        # model parameter değildir,
        # ama model ile birlikte device'a taşınır.
        self.register_buffer(
            "causal_mask",
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        #
        # [B, T, C]

        batch_size, sequence_length, channels = (
            x.shape
        )

        # ------------------------------------------
        # Q K V
        # ------------------------------------------

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # q, k, v:
        #
        # [B, T, HEAD_SIZE]

        # ------------------------------------------
        # Attention scores
        # ------------------------------------------

        # q:
        # [B, T, H]
        #
        # k.transpose:
        # [B, H, T]
        #
        # sonuç:
        # [B, T, T]

        attention_scores = (
            q @ k.transpose(-2, -1)
        )

        # Scaled dot-product attention
        attention_scores = (
            attention_scores
            / math.sqrt(self.head_size)
        )

        # ------------------------------------------
        # Causal mask
        # ------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = (
            attention_scores.masked_fill(
                ~mask,
                float("-inf"),
            )
        )

        # ------------------------------------------
        # Softmax
        # ------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # ------------------------------------------
        # Weighted sum of values
        # ------------------------------------------

        # [B,T,T]
        # @
        # [B,T,H]
        #
        # =
        #
        # [B,T,H]

        output = (
            attention_weights @ v
        )

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # 6 farklı attention head.
        #
        # Her biri:
        #
        # 384 -> 64
        #
        # 6 × 64 = 384
        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    head_size=HEAD_SIZE
                )
                for _ in range(6)
            ]
        )

        # Head çıktıları concatenate edildikten sonra
        # tekrar embedding uzayına projekte ediyoruz.
        self.projection = nn.Linear(
            N_EMBD,
            N_EMBD,
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Her head:
        #
        # [B, T, 384]
        #       ↓
        # [B, T, 64]
        #
        # 6 head concatenate:
        #
        # [B, T, 384]

        output = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1,
        )

        output = self.projection(
            output
        )

        output = self.dropout(
            output
        )

        return output