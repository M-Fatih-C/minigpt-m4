from pathlib import Path

import numpy as np

from datasets import load_dataset

from tokenizer.bpe import BPETokenizer


DATASET_NAME = "roneneldan/TinyStories"

TOKENIZER_PATH = Path(
    "tokenizer/tokenizer.json"
)

TRAIN_OUTPUT = Path(
    "data/train_v2.bin"
)

VALID_OUTPUT = Path(
    "data/validation_v2.bin"
)

# İlk datasetimizle yaklaşık aynı story sayıları.
TRAIN_STORIES = 289_234
VALIDATION_STORIES = 21_990

END_OF_STORY = "\n<|END_OF_STORY|>\n"

FLUSH_TOKEN_COUNT = 1_000_000


def flush_buffer(
    buffer: list[int],
    file,
) -> int:

    if not buffer:
        return 0

    array = np.asarray(
        buffer,
        dtype=np.uint16,
    )

    array.tofile(file)

    count = len(array)

    buffer.clear()

    return count


def encode_split(
    tokenizer: BPETokenizer,
    split: str,
    output_path: Path,
    max_stories: int,
):

    print()
    print("=" * 60)
    print(f"Encoding TinyStories v2: {split}")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    eos_tokens = tokenizer.encode(
        END_OF_STORY
    )

    print(
        f"EOS marker tokens : "
        f"{len(eos_tokens)}"
    )

    print(
        f"EOS token IDs     : "
        f"{eos_tokens}"
    )

    print()

    buffer = []

    story_count = 0
    total_tokens = 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "wb"
    ) as destination:

        for sample in dataset:

            story = sample["text"].strip()

            if not story:
                continue

            story_tokens = tokenizer.encode(
                story
            )

            buffer.extend(
                story_tokens
            )

            buffer.extend(
                eos_tokens
            )

            story_count += 1

            if (
                len(buffer)
                >= FLUSH_TOKEN_COUNT
            ):

                total_tokens += (
                    flush_buffer(
                        buffer,
                        destination,
                    )
                )

            if (
                story_count % 10_000
                == 0
            ):

                print(
                    f"\rStories: "
                    f"{story_count:>8,}"
                    f" | tokens: "
                    f"{total_tokens + len(buffer):>12,}",
                    end="",
                    flush=True,
                )

            if story_count >= max_stories:
                break

        total_tokens += flush_buffer(
            buffer,
            destination,
        )

    print()
    print()
    print(
        f"Stories    : {story_count:,}"
    )

    print(
        f"Tokens     : {total_tokens:,}"
    )

    print(
        f"File size  : "
        f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    return (
        story_count,
        total_tokens,
        eos_tokens,
    )


def count_sequence(
    data: np.memmap,
    sequence: list[int],
) -> int:

    """
    EOS marker'ın gerçekten dataset içinde
    beklendiği kadar bulunduğunu doğrular.

    İlk token üzerinden aday noktaları bulup
    tüm sequence'i kontrol ediyoruz.
    """

    first_token = sequence[0]

    candidates = np.where(
        data == first_token
    )[0]

    count = 0

    sequence_array = np.asarray(
        sequence,
        dtype=np.uint16,
    )

    length = len(sequence)

    for index in candidates:

        if index + length > len(data):
            continue

        if np.array_equal(
            data[
                index:
                index + length
            ],
            sequence_array,
        ):
            count += 1

    return count


def verify(
    path: Path,
    eos_tokens: list[int],
    expected_stories: int,
):

    print()
    print(f"Verifying {path}...")

    data = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

    eos_count = count_sequence(
        data,
        eos_tokens,
    )

    print(
        f"Total tokens : {len(data):,}"
    )

    print(
        f"EOS markers  : {eos_count:,}"
    )

    print(
        f"Expected     : "
        f"{expected_stories:,}"
    )

    if eos_count != expected_stories:

        raise RuntimeError(
            "EOS verification failed."
        )

    print(
        "EOS verification passed ✓"
    )


def main():

    print("=" * 60)
    print("MiniGPT-M4 Dataset v2")
    print("=" * 60)

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    print(
        f"Tokenizer vocab: "
        f"{tokenizer.vocab_size:,}"
    )

    print(
        f"EOS marker: "
        f"{repr(END_OF_STORY)}"
    )

    train_stories, _, eos_tokens = (
        encode_split(
            tokenizer,
            split="train",
            output_path=TRAIN_OUTPUT,
            max_stories=TRAIN_STORIES,
        )
    )

    validation_stories, _, _ = (
        encode_split(
            tokenizer,
            split="validation",
            output_path=VALID_OUTPUT,
            max_stories=VALIDATION_STORIES,
        )
    )

    print()
    print("=" * 60)
    print("VERIFY")
    print("=" * 60)

    verify(
        TRAIN_OUTPUT,
        eos_tokens,
        train_stories,
    )

    verify(
        VALID_OUTPUT,
        eos_tokens,
        validation_stories,
    )

    print()
    print("=" * 60)
    print("DATASET V2 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()