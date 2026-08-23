import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v3 import (
    BLOCK_SIZE,
    N_EMBD,
    N_LAYER,
    VOCAB_SIZE,
)

from model_v3.block import (
    TransformerBlock,
)

from model_v3.norm import RMSNorm


class MiniGPTV3(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            N_EMBD,
        )

        # ----------------------------------------------------
        # Transformer stack
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(N_LAYER)
            ]
        )

        # ----------------------------------------------------
        # Final RMSNorm
        # ----------------------------------------------------

        self.final_norm = RMSNorm(
            N_EMBD
        )

        # ----------------------------------------------------
        # LM head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            N_EMBD,
            VOCAB_SIZE,
            bias=False,
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self.apply(
            self._init_weights
        )

        self._scale_residual_weights()

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    @staticmethod
    def _init_weights(
        module: nn.Module,
    ) -> None:

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def _scale_residual_weights(
        self,
    ) -> None:
        """
        Residual output projections receive a smaller
        initialization as depth increases.
        """

        scale = (
            0.02
            / math.sqrt(
                2 * N_LAYER
            )
        )

        for block in self.blocks:

            nn.init.normal_(
                block
                .attention
                .output_projection
                .weight,
                mean=0.0,
                std=scale,
            )

            nn.init.normal_(
                block
                .feed_forward
                .down_projection
                .weight,
                mean=0.0,
                std=scale,
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:

        _, sequence_length = (
            token_ids.shape
        )

        if sequence_length > BLOCK_SIZE:

            raise ValueError(
                f"Sequence length "
                f"{sequence_length} exceeds "
                f"BLOCK_SIZE={BLOCK_SIZE}"
            )

        # ----------------------------------------------------
        # Embedding
        #
        # No learned positional embedding.
        # Position information is introduced through RoPE.
        # ----------------------------------------------------

        x = self.token_embedding(
            token_ids
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        x = self.final_norm(
            x
        )

        logits = self.lm_head(
            x
        )

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    VOCAB_SIZE,
                ),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        for _ in range(
            max_new_tokens
        ):

            context = token_ids[
                :,
                -BLOCK_SIZE:
            ]

            logits, _ = self(
                context
            )

            logits = (
                logits[:, -1, :]
                / temperature
            )

            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                cutoff = values[
                    :,
                    -1:
                ]

                logits = (
                    logits.masked_fill(
                        logits < cutoff,
                        float("-inf"),
                    )
                )

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            next_token = (
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                )
            )

            token_ids = torch.cat(
                (
                    token_ids,
                    next_token,
                ),
                dim=1,
            )

        return token_ids
