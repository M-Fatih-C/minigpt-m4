import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v3 import (
    DROPOUT,
    HEAD_SIZE,
    N_EMBD,
    N_HEAD,
)

from model_v3.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.n_head = N_HEAD
        self.head_size = HEAD_SIZE
        self.n_embd = N_EMBD

        # ----------------------------------------------------
        # Fused QKV projection
        # ----------------------------------------------------

        self.qkv_projection = nn.Linear(
            N_EMBD,
            3 * N_EMBD,
            bias=False,
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.output_projection = nn.Linear(
            N_EMBD,
            N_EMBD,
            bias=False,
        )

        self.rope = RotaryEmbedding()

        self.dropout = DROPOUT

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = (
            x.shape
        )

        # ----------------------------------------------------
        # Q K V
        #
        # [B,T,C]
        #     ↓
        # [B,T,3C]
        # ----------------------------------------------------

        qkv = self.qkv_projection(
            x
        )

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        # ----------------------------------------------------
        # Multi-head reshape
        #
        # [B,T,C]
        #     ↓
        # [B,H,T,D]
        # ----------------------------------------------------

        q = (
            q.view(
                batch_size,
                sequence_length,
                self.n_head,
                self.head_size,
            )
            .transpose(1, 2)
        )

        k = (
            k.view(
                batch_size,
                sequence_length,
                self.n_head,
                self.head_size,
            )
            .transpose(1, 2)
        )

        v = (
            v.view(
                batch_size,
                sequence_length,
                self.n_head,
                self.head_size,
            )
            .transpose(1, 2)
        )

        # ----------------------------------------------------
        # Rotary positional embeddings
        # ----------------------------------------------------

        q, k = self.rope(
            q,
            k,
        )

        # ----------------------------------------------------
        # Optimized scaled dot-product attention
        # ----------------------------------------------------

        dropout_p = (
            self.dropout
            if self.training
            else 0.0
        )

        output = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        )

        # ----------------------------------------------------
        # [B,H,T,D]
        #     ↓
        # [B,T,C]
        # ----------------------------------------------------

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.n_embd,
            )
        )

        output = self.output_projection(
            output
        )

        return output
