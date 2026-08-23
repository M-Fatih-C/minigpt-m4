from pathlib import Path

import numpy as np
import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
)


DATA_DIR = Path("data")

IGNORE_INDEX = -100


class SmolTalkDataset:
    """
    Conversation-aware SmolTalk dataset.

    Her example disk üzerinde ayrı tutulur:

        offsets[i]
            ↓
        example i
            ↓
        offsets[i + 1]

    Böylece DailyDialog denememizdeki gibi
    rastgele bir conversation'ın ortasından
    batch başlatmıyoruz.
    """

    def __init__(
        self,
        split: str,
    ):
        if split not in (
            "train",
            "validation",
        ):
            raise ValueError(
                f"Unknown split: {split}"
            )

        self.split = split

        prefix = (
            DATA_DIR
            / f"smoltalk_{split}"
        )

        token_path = Path(
            f"{prefix}_tokens.bin"
        )

        mask_path = Path(
            f"{prefix}_mask.bin"
        )

        offset_path = Path(
            f"{prefix}_offsets.npy"
        )

        # --------------------------------------------------
        # Existence checks
        # --------------------------------------------------

        for path in (
            token_path,
            mask_path,
            offset_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing dataset file: {path}"
                )

        # --------------------------------------------------
        # Memory mapped data
        # --------------------------------------------------

        self.tokens = np.memmap(
            token_path,
            dtype=np.uint16,
            mode="r",
        )

        self.masks = np.memmap(
            mask_path,
            dtype=np.uint8,
            mode="r",
        )

        self.offsets = np.load(
            offset_path,
            mmap_mode="r",
        )

        self.example_count = (
            len(self.offsets) - 1
        )

        # --------------------------------------------------
        # Sanity checks
        # --------------------------------------------------

        if (
            len(self.tokens)
            != len(self.masks)
        ):
            raise RuntimeError(
                "Token/mask length mismatch."
            )

        if self.example_count <= 0:
            raise RuntimeError(
                "Dataset contains no examples."
            )

        if self.offsets[0] != 0:
            raise RuntimeError(
                "First offset must be zero."
            )

        if (
            self.offsets[-1]
            != len(self.tokens)
        ):
            raise RuntimeError(
                "Final offset does not match token count."
            )

        lengths = np.diff(
            self.offsets
        )

        if lengths.max() > (
            BLOCK_SIZE + 1
        ):
            raise RuntimeError(
                "Dataset contains an example "
                "longer than BLOCK_SIZE + 1."
            )

        self.min_length = int(
            lengths.min()
        )

        self.max_length = int(
            lengths.max()
        )

        self.avg_length = float(
            lengths.mean()
        )

        self.target_count = int(
            self.masks.sum()
        )

        print(
            f"SmolTalk dataset [{split}]"
        )

        print(
            f"Examples       : "
            f"{self.example_count:,}"
        )

        print(
            f"Tokens         : "
            f"{len(self.tokens):,}"
        )

        print(
            f"Target tokens  : "
            f"{self.target_count:,}"
        )

        print(
            f"Length range   : "
            f"{self.min_length} - "
            f"{self.max_length}"
        )

        print(
            f"Average length : "
            f"{self.avg_length:.1f}"
        )

    # ======================================================
    # Individual example
    # ======================================================

    def get_example(
        self,
        example_id: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Raw tokens + masks döndürür.
        """

        if not (
            0
            <= example_id
            < self.example_count
        ):
            raise IndexError(
                f"Example ID out of range: "
                f"{example_id}"
            )

        start = int(
            self.offsets[
                example_id
            ]
        )

        end = int(
            self.offsets[
                example_id + 1
            ]
        )

        tokens = np.asarray(
            self.tokens[
                start:end
            ],
            dtype=np.int64,
        )

        masks = np.asarray(
            self.masks[
                start:end
            ],
            dtype=np.uint8,
        )

        return tokens, masks

    # ======================================================
    # Training batch
    # ======================================================

    def get_batch(
        self,
        device: torch.device,
        batch_size: int = BATCH_SIZE,
        block_size: int = BLOCK_SIZE,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        GPT batch üretir.

        Example:

            token0 token1 token2 token3

        X:

            token0 token1 token2

        Y:

            token1 token2 token3

        Mask, Y ile aynı şekilde +1 kaydırılır.

        Assistant dışındaki targetlar:

            -100

        yapılarak CrossEntropyLoss tarafından
        ignore edilir.
        """

        # Padding için 0 kullanıyoruz.
        #
        # Padding pozisyonlarının Y targetı -100 olduğu
        # için loss hesaplanmaz.
        #
        # Causal attention sayesinde gerçek tokenlar
        # gelecekteki padding tokenlarını göremez.
        x_batch = np.zeros(
            (
                batch_size,
                block_size,
            ),
            dtype=np.int64,
        )

        y_batch = np.full(
            (
                batch_size,
                block_size,
            ),
            fill_value=IGNORE_INDEX,
            dtype=np.int64,
        )

        # Her batch row için bağımsız bir
        # conversation-aware example seç.
        example_ids = np.random.randint(
            low=0,
            high=self.example_count,
            size=batch_size,
        )

        for row_index, example_id in enumerate(
            example_ids
        ):
            tokens, masks = (
                self.get_example(
                    int(example_id)
                )
            )

            # En az iki token lazım:
            #
            # X = tokens[:-1]
            # Y = tokens[1:]
            if len(tokens) < 2:
                continue

            x = tokens[:-1]

            y = tokens[1:].copy()

            # y[t], x[t]'den tahmin edildiği için
            # supervision mask de bir token ileri kayar.
            target_mask = (
                masks[1:]
            )

            length = min(
                len(x),
                block_size,
            )

            # --------------------------------------------------
            # Input
            # --------------------------------------------------

            x_batch[
                row_index,
                :length
            ] = x[:length]

            # --------------------------------------------------
            # Targets
            # --------------------------------------------------

            targets = (
                y[:length].copy()
            )

            targets[
                target_mask[:length]
                == 0
            ] = IGNORE_INDEX

            y_batch[
                row_index,
                :length
            ] = targets

        # ------------------------------------------------------
        # NumPy → Torch → MPS
        # ------------------------------------------------------

        x_tensor = torch.from_numpy(
            x_batch
        ).to(device)

        y_tensor = torch.from_numpy(
            y_batch
        ).to(device)

        return (
            x_tensor,
            y_tensor,
        )


# ============================================================
# Standalone sanity test
# ============================================================

if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("MiniGPT-M4 SmolTalk Dataset Test")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    print()

    train_dataset = SmolTalkDataset(
        "train"
    )

    print()

    validation_dataset = (
        SmolTalkDataset(
            "validation"
        )
    )

    print()
    print("=" * 60)
    print("BATCH TEST")
    print("=" * 60)

    x, y = train_dataset.get_batch(
        device=device
    )

    print()
    print(
        f"X shape : {x.shape}"
    )

    print(
        f"Y shape : {y.shape}"
    )

    print(
        f"X dtype : {x.dtype}"
    )

    print(
        f"Y dtype : {y.dtype}"
    )

    print(
        f"X device: {x.device}"
    )

    print(
        f"Y device: {y.device}"
    )

    supervised = (
        y != IGNORE_INDEX
    ).sum().item()

    ignored = (
        y == IGNORE_INDEX
    ).sum().item()

    total = y.numel()

    print()
    print(
        f"Supervised targets : "
        f"{supervised:,}"
    )

    print(
        f"Ignored targets    : "
        f"{ignored:,}"
    )

    print(
        f"Supervised ratio   : "
        f"{100 * supervised / total:.2f}%"
    )

    # ------------------------------------------------------
    # Assertions
    # ------------------------------------------------------

    assert x.shape == (
        BATCH_SIZE,
        BLOCK_SIZE,
    )

    assert y.shape == (
        BATCH_SIZE,
        BLOCK_SIZE,
    )

    assert x.dtype == torch.int64

    assert y.dtype == torch.int64

    assert supervised > 0

    assert ignored > 0

    print()
    print(
        "SmolTalk batch test passed ✓"
    )