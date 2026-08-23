import numpy as np
import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
)


class ChatDataset:

    def __init__(
        self,
        split="train",
    ):

        token_path = (
            f"data/dialog_{split}.bin"
        )

        mask_path = (
            f"data/dialog_{split}_mask.bin"
        )

        self.tokens = np.memmap(
            token_path,
            dtype=np.uint16,
            mode="r",
        )

        self.mask = np.memmap(
            mask_path,
            dtype=np.uint8,
            mode="r",
        )

        assert (
            len(self.tokens)
            == len(self.mask)
        )

        print(
            f"Chat dataset [{split}]"
        )

        print(
            f"Tokens            : "
            f"{len(self.tokens):,}"
        )

        print(
            f"Assistant targets : "
            f"{int(self.mask.sum()):,}"
        )

    def get_batch(
        self,
        device,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        min_targets=16,
    ):

        xs = []
        ys = []

        max_start = (
            len(self.tokens)
            - block_size
            - 1
        )

        while len(xs) < batch_size:

            start = np.random.randint(
                0,
                max_start,
            )

            x = np.asarray(
                self.tokens[
                    start:
                    start + block_size
                ],
                dtype=np.int64,
            )

            y = np.asarray(
                self.tokens[
                    start + 1:
                    start + block_size + 1
                ],
                dtype=np.int64,
            )

            # Önemli:
            #
            # y[t], x[t]'den tahmin ediliyor.
            #
            # Dolayısıyla mask de +1 kaydırılmalı.
            target_mask = np.asarray(
                self.mask[
                    start + 1:
                    start + block_size + 1
                ],
                dtype=np.uint8,
            )

            if (
                target_mask.sum()
                < min_targets
            ):
                continue

            # USER bölümlerini loss'tan çıkar.
            y[
                target_mask == 0
            ] = -100

            xs.append(x)
            ys.append(y)

        x = torch.tensor(
            np.stack(xs),
            dtype=torch.long,
            device=device,
        )

        y = torch.tensor(
            np.stack(ys),
            dtype=torch.long,
            device=device,
        )

        return x, y