from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"

OUTPUT_DIR = Path("data/pretrain")

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "fineweb_edu_train.txt"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "fineweb_edu_validation.txt"
)


# 2.5 GiB raw training text.
TRAIN_TARGET_BYTES = int(
    2.5 * 1024**3
)

# 64 MiB validation.
VALIDATION_TARGET_BYTES = (
    64 * 1024**2
)

SHUFFLE_BUFFER = 10_000
SEED = 42

END_OF_DOCUMENT = (
    "\n<|END_OF_TEXT|>\n"
)


# ============================================================
# Helpers
# ============================================================

def format_document(
    text: str,
) -> str:

    return (
        text.strip()
        + END_OF_DOCUMENT
    )


def size_mib(
    byte_count: int,
) -> float:

    return (
        byte_count
        / 1024**2
    )


def size_gib(
    byte_count: int,
) -> float:

    return (
        byte_count
        / 1024**3
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "MiniGPT-M4 V3 FineWeb-Edu Preparation"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset       : "
        f"{DATASET_NAME}"
    )

    print(
        f"Config        : "
        f"{DATASET_CONFIG}"
    )

    print(
        f"Train target  : "
        f"{size_gib(TRAIN_TARGET_BYTES):.2f} GiB"
    )

    print(
        f"Val target    : "
        f"{size_mib(VALIDATION_TARGET_BYTES):.0f} MiB"
    )

    print()
    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    # Streaming shuffle.
    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER,
    )

    train_bytes = 0
    validation_bytes = 0

    train_documents = 0
    validation_documents = 0

    skipped_documents = 0

    total_documents = 0

    with (
        TRAIN_OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as train_file,

        VALIDATION_OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as validation_file,
    ):

        progress = tqdm(
            desc="FineWeb-Edu",
            unit="doc",
        )

        for row in dataset:

            total_documents += 1

            text = row.get(
                "text"
            )

            if (
                not isinstance(text, str)
                or not text.strip()
            ):

                skipped_documents += 1
                continue

            document = (
                format_document(
                    text
                )
            )

            document_bytes = len(
                document.encode(
                    "utf-8"
                )
            )

            # ------------------------------------------------
            # First fill validation.
            # ------------------------------------------------

            if (
                validation_bytes
                < VALIDATION_TARGET_BYTES
                and
                total_documents % 50 == 0
            ):

                validation_file.write(
                    document
                )

                validation_bytes += (
                    document_bytes
                )

                validation_documents += 1

            # ------------------------------------------------
            # Everything else → train.
            # ------------------------------------------------

            elif (
                train_bytes
                < TRAIN_TARGET_BYTES
            ):

                train_file.write(
                    document
                )

                train_bytes += (
                    document_bytes
                )

                train_documents += 1

            progress.update(1)

            progress.set_postfix(
                train=(
                    f"{size_gib(train_bytes):.2f}G"
                ),
                val=(
                    f"{size_mib(validation_bytes):.0f}M"
                ),
            )

            if (
                train_bytes
                >= TRAIN_TARGET_BYTES
                and
                validation_bytes
                >= VALIDATION_TARGET_BYTES
            ):
                break

        progress.close()

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()
    print(
        f"Documents seen     : "
        f"{total_documents:,}"
    )

    print(
        f"Skipped documents  : "
        f"{skipped_documents:,}"
    )

    print()
    print(
        f"Train documents    : "
        f"{train_documents:,}"
    )

    print(
        f"Train size         : "
        f"{size_gib(train_bytes):.3f} GiB"
    )

    print()
    print(
        f"Validation docs    : "
        f"{validation_documents:,}"
    )

    print(
        f"Validation size    : "
        f"{size_mib(validation_bytes):.2f} MiB"
    )

    print()
    print(
        f"Train file         : "
        f"{TRAIN_OUTPUT}"
    )

    print(
        f"Validation file    : "
        f"{VALIDATION_OUTPUT}"
    )

    print()
    print("=" * 70)
    print(
        "FINEWEB-EDU PREPARATION COMPLETE ✓"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
