from pathlib import Path

import numpy as np
import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
)


class TokenDataset:

    def __init__(
        self,
        train_path: str = TRAIN_DATA_PATH,
        validation_path: str = VALID_DATA_PATH,
    ):

        self.train = np.memmap(
            train_path,
            dtype=np.uint16,
            mode="r",
        )

        self.validation = np.memmap(
            validation_path,
            dtype=np.uint16,
            mode="r",
        )

        print("Dataset loaded.")
        print(
            f"Train tokens      : "
            f"{len(self.train):,}"
        )
        print(
            f"Validation tokens : "
            f"{len(self.validation):,}"
        )

    def get_batch(
        self,
        split: str,
        device: torch.device,
        batch_size: int = BATCH_SIZE,
        block_size: int = BLOCK_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if split == "train":
            data = self.train

        elif split in ("validation", "val"):
            data = self.validation

        else:
            raise ValueError(
                f"Unknown split: {split}"
            )

        # Her batch elemanı için dataset içerisinde
        # random başlangıç noktası seçiyoruz.
        starts = torch.randint(
            low=0,
            high=len(data) - block_size - 1,
            size=(batch_size,),
        ).tolist()

        # X:
        #
        # token[i]
        # token[i+1]
        # ...
        # token[i+255]
        #
        x_numpy = np.stack(
            [
                data[
                    start:
                    start + block_size
                ]
                for start in starts
            ]
        ).astype(np.int64)

        # Y aynı sequence'in 1 token ileri kaydırılmış hali.
        #
        # token[i+1]
        # token[i+2]
        # ...
        # token[i+256]
        #
        y_numpy = np.stack(
            [
                data[
                    start + 1:
                    start + block_size + 1
                ]
                for start in starts
            ]
        ).astype(np.int64)

        x = torch.from_numpy(
            x_numpy
        ).to(device)

        y = torch.from_numpy(
            y_numpy
        ).to(device)

        return x, y