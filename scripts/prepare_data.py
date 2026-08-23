from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATA_DIR = Path("data")

TRAIN_TARGET_MB = 250
VALID_TARGET_MB = 20

DATASET_NAME = "roneneldan/TinyStories"


def prepare_split(
    split: str,
    output_path: Path,
    target_mb: int,
) -> None:
    """
    TinyStories datasetini streaming olarak okur.

    target_mb kadar UTF-8 text topladığında durur.
    Böylece datasetin tamamını indirmemize gerek kalmaz.
    """

    print(f"\nLoading split: {split}")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    target_bytes = target_mb * 1024 * 1024
    written_bytes = 0
    story_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(
        total=target_bytes,
        unit="B",
        unit_scale=True,
        desc=split,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for sample in dataset:

            story = sample["text"].strip()

            if not story:
                continue

            # Hikâyeleri birbirinden ayırıyoruz.
            text = story + "\n\n"

            encoded = text.encode("utf-8")
            byte_count = len(encoded)

            output_file.write(text)

            written_bytes += byte_count
            story_count += 1

            progress.update(byte_count)

            if written_bytes >= target_bytes:
                break

    progress.close()

    print(f"\nCompleted: {split}")
    print(f"Stories     : {story_count:,}")
    print(
        f"Size        : "
        f"{written_bytes / 1024 / 1024:.2f} MB"
    )
    print(f"Output      : {output_path}")


def main() -> None:

    DATA_DIR.mkdir(exist_ok=True)

    prepare_split(
        split="train",
        output_path=DATA_DIR / "train.txt",
        target_mb=TRAIN_TARGET_MB,
    )

    prepare_split(
        split="validation",
        output_path=DATA_DIR / "validation.txt",
        target_mb=VALID_TARGET_MB,
    )

    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()