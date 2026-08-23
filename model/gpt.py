import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BLOCK_SIZE,
    N_EMBD,
    N_LAYER,
    VOCAB_SIZE,
)

from model.block import TransformerBlock
from model.embedding import GPTEmbedding


class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # Input embedding
        # --------------------------------------------------

        self.embedding = GPTEmbedding()

        # --------------------------------------------------
        # Transformer stack
        # --------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(N_LAYER)
            ]
        )

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        self.final_norm = nn.LayerNorm(
            N_EMBD
        )

        # --------------------------------------------------
        # Language-model head
        # --------------------------------------------------

        self.lm_head = nn.Linear(
            N_EMBD,
            VOCAB_SIZE,
            bias=False,
        )

        # Ağırlıkları başlat.
        self.apply(
            self._init_weights
        )

        # --------------------------------------------------
        # Weight tying
        # --------------------------------------------------
        #
        # Input:
        #
        # token id
        #   ↓
        # [2048 x 384]
        #
        # Output:
        #
        # hidden state
        #   ↓
        # [384 x 2048]
        #
        # Aynı semantic token space'i kullandığımız için
        # iki matrisi paylaştırıyoruz.

        self.lm_head.weight = (
            self.embedding
            .token_embedding
            .weight
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

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
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

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:

        # token_ids:
        #
        # [B, T]

        batch_size, sequence_length = (
            token_ids.shape
        )

        if sequence_length > BLOCK_SIZE:

            raise ValueError(
                f"Sequence length "
                f"{sequence_length} exceeds "
                f"BLOCK_SIZE={BLOCK_SIZE}"
            )

        # --------------------------------------------------
        # Embedding
        # --------------------------------------------------

        # [B,T]
        #
        # ↓
        #
        # [B,T,384]

        x = self.embedding(
            token_ids
        )

        # --------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # --------------------------------------------------
        # Final LayerNorm
        # --------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------
        # Vocabulary logits
        # --------------------------------------------------

        # [B,T,384]
        #
        # ↓
        #
        # [B,T,2048]

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            # CrossEntropyLoss:
            #
            # logits:
            # [B,T,V]
            #
            # targets:
            # [B,T]
            #
            # flatten:
            #
            # logits  -> [B*T, V]
            # targets -> [B*T]

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

            # Context model limitini aşarsa
            # yalnızca son 256 tokenı kullan.
            context = token_ids[
                :,
                -BLOCK_SIZE:
            ]

            logits, _ = self(
                context
            )

            # Sadece son pozisyonun
            # next-token tahmini gerekiyor.
            logits = (
                logits[:, -1, :]
                / temperature
            )

            # İsteğe bağlı top-k sampling.
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
                    -1
                ].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
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