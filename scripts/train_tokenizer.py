from pathlib import Path

from tokenizer.bpe import BPETokenizer


TRAIN_FILE = Path("data/train.txt")
VALIDATION_FILE = Path("data/validation.txt")

TOKENIZER_PATH = Path(
    "tokenizer/tokenizer.json"
)

# BPE öğrenmek için bütün 250 MB'a ihtiyacımız yok.
TOKENIZER_TRAIN_BYTES = 2 * 1024 * 1024

VOCAB_SIZE = 2048


def read_text_prefix(
    path: Path,
    max_bytes: int,
) -> str:

    with path.open("rb") as file:
        data = file.read(max_bytes)

    return data.decode(
        "utf-8",
        errors="ignore",
    )


def main():

    print("=" * 60)
    print("MiniGPT-M4 BPE Tokenizer")
    print("=" * 60)

    print()
    print(
        f"Reading tokenizer training sample "
        f"from {TRAIN_FILE}..."
    )

    training_text = read_text_prefix(
        TRAIN_FILE,
        TOKENIZER_TRAIN_BYTES,
    )

    print(
        f"Loaded: "
        f"{len(training_text.encode('utf-8')) / 1024 / 1024:.2f} MB"
    )

    print()

    tokenizer = BPETokenizer.train(
        training_text,
        vocab_size=VOCAB_SIZE,
        min_pair_frequency=2,
        log_every=100,
    )

    tokenizer.save(
        TOKENIZER_PATH
    )

    print()
    print(
        f"Tokenizer saved to: "
        f"{TOKENIZER_PATH}"
    )

    print()
    print("=" * 60)
    print("ROUNDTRIP TEST")
    print("=" * 60)

    examples = [
        "Once upon a time, there was a little dragon.",
        "Lily was very happy to see her friend.",
        "Hello world!",
        "The little cat ran through the garden.",
    ]

    for text in examples:

        token_ids = tokenizer.encode(text)

        decoded = tokenizer.decode(
            token_ids
        )

        print()
        print("Text   :", text)
        print("Tokens :", token_ids)
        print("Count  :", len(token_ids))
        print("Decoded:", decoded)

        assert decoded == text

    print()
    print("Roundtrip tests passed ✓")

    print()
    print("=" * 60)
    print("COMPRESSION TEST")
    print("=" * 60)

    validation_sample = read_text_prefix(
        VALIDATION_FILE,
        1024 * 1024,
    )

    validation_tokens = tokenizer.encode(
        validation_sample
    )

    byte_count = len(
        validation_sample.encode("utf-8")
    )

    token_count = len(
        validation_tokens
    )

    bytes_per_token = (
        byte_count / token_count
    )

    print(
        f"Validation bytes : "
        f"{byte_count:,}"
    )

    print(
        f"Token count      : "
        f"{token_count:,}"
    )

    print(
        f"Bytes / token    : "
        f"{bytes_per_token:.2f}"
    )

    print()

    assert (
        tokenizer.decode(validation_tokens)
        == validation_sample
    )

    print(
        "Validation roundtrip passed ✓"
    )

    print()
    print(
        f"Final vocab size: "
        f"{tokenizer.vocab_size:,}"
    )


if __name__ == "__main__":
    main()