import torch
import torch.nn as nn

from config_v3 import (
    BLOCK_SIZE,
    HEAD_SIZE,
    ROPE_BASE,
)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int = HEAD_SIZE,
        max_seq_len: int = BLOCK_SIZE,
        base: float = ROPE_BASE,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE head dimension must be even."
            )

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inverse_frequency = (
            1.0
            / (
                base
                ** (
                    torch.arange(
                        0,
                        head_dim,
                        2,
                        dtype=torch.float32,
                    )
                    / head_dim
                )
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        frequencies = torch.outer(
            positions,
            inverse_frequency,
        )

        self.register_buffer(
            "cos_cache",
            frequencies.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            frequencies.sin(),
            persistent=False,
        )

    def apply_rotary(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B, H, T, D]
        """

        sequence_length = x.size(-2)

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length "
                f"{sequence_length} exceeds "
                f"RoPE limit "
                f"{self.max_seq_len}"
            )

        cos = self.cos_cache[
            :sequence_length
        ]

        sin = self.sin_cache[
            :sequence_length
        ]

        cos = cos[
            None,
            None,
            :,
            :
        ]

        sin = sin[
            None,
            None,
            :,
            :
        ]

        cos = cos.to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = sin.to(
            device=x.device,
            dtype=x.dtype,
        )

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = (
            even * cos
            - odd * sin
        )

        rotated_odd = (
            even * sin
            + odd * cos
        )

        rotated = torch.stack(
            (
                rotated_even,
                rotated_odd,
            ),
            dim=-1,
        )

        return rotated.flatten(
            start_dim=-2
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.apply_rotary(q),
            self.apply_rotary(k),
        )
