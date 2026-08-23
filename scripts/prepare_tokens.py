from pathlib import Path

import numpy as np

from tokenizer.bpe import (
    BPETokenizer,
    PRETOKEN_PATTERN,
)


TOKENIZER_PATH = Path("tokenizer/tokenizer.json")

TRAIN_TEXT = Path("data/train.txt")
VALID_TEXT = Path("data/validation.txt")

TRAIN_BIN = Path("data/train.bin")
VALID_BIN = Path("data/validation.bin")

# Metni komple RAM'e almak istemiyoruz.
CHUNK_SIZE = 1024 * 1024  # ~1 MB karakter


def encode_file(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
) -> None:

    total_tokens = 0
    total_chars = 0

    # Chunk sınırında bir kelimeyi veya whitespace grubunu
    # ikiye bölmemek için son pretoken'ı sonraki chunk'a taşıyoruz.
    carry = ""

    print()
    print("=" * 60)
    print(f"Encoding: {input_path}")
    print("=" * 60)

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as source,
        output_path.open(
            "wb",
        ) as destination,
    ):

        while True:

            chunk = source.read(CHUNK_SIZE)

            if not chunk:
                break

            total_chars += len(chunk)

            text = carry + chunk

            pieces = PRETOKEN_PATTERN.findall(text)

            if not pieces:
                carry = text
                continue

            # Son parça chunk sınırında yarım kalmış olabilir.
            # Bir sonraki chunk ile tekrar değerlendireceğiz.
            carry = pieces[-1]

            complete_text = "".join(
                pieces[:-1]
            )

            if not complete_text:
                continue

            token_ids = tokenizer.encode(
                complete_text
            )

            array = np.asarray(
                token_ids,
                dtype=np.uint16,
            )

            array.tofile(destination)

            total_tokens += len(array)

            print(
                f"\rCharacters: {total_chars:>12,}"
                f" | Tokens: {total_tokens:>12,}",
                end="",
                flush=True,
            )

        # EOF geldi. Artık carry'nin kesinlikle tamamı elimizde.
        if carry:

            token_ids = tokenizer.encode(carry)

            array = np.asarray(
                token_ids,
                dtype=np.uint16,
            )

            array.tofile(destination)

            total_tokens += len(array)

    print()
    print()
    print(f"Completed: {output_path}")
    print(f"Characters : {total_chars:,}")
    print(f"Tokens     : {total_tokens:,}")
    print(
        f"Bin size   : "
        f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    if total_tokens > 0:

        bytes_in_source = (
            input_path.stat().st_size
        )

        print(
            f"Bytes/token: "
            f"{bytes_in_source / total_tokens:.3f}"
        )


def verify_dataset(
    tokenizer: BPETokenizer,
    text_path: Path,
    bin_path: Path,
) -> None:

    """
    Datasetin başlangıcından küçük bir örnek alıp
    tokenları tekrar metne çeviriyoruz.
    """

    tokens = np.memmap(
        bin_path,
        dtype=np.uint16,
        mode="r",
    )

    sample_tokens = tokens[:500]

    decoded = tokenizer.decode(
        sample_tokens.tolist()
    )

    with text_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        original_start = file.read(
            len(decoded)
        )

    if decoded != original_start:

        raise RuntimeError(
            f"Verification failed for {bin_path}"
        )

    print(
        f"Verification passed ✓ "
        f"({bin_path})"
    )


def main():

    print("=" * 60)
    print("MiniGPT-M4 Dataset Tokenization")
    print("=" * 60)

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    print(
        f"Tokenizer vocab size: "
        f"{tokenizer.vocab_size:,}"
    )

    # uint16 maksimum 65535.
    # Bizim vocab 2048 olduğu için fazlasıyla yeterli.
    if tokenizer.vocab_size > 65535:
        raise ValueError(
            "Vocabulary does not fit inside uint16."
        )

    encode_file(
        tokenizer,
        TRAIN_TEXT,
        TRAIN_BIN,
    )

    encode_file(
        tokenizer,
        VALID_TEXT,
        VALID_BIN,
    )

    print()
    print("=" * 60)
    print("VERIFY")
    print("=" * 60)

    verify_dataset(
        tokenizer,
        TRAIN_TEXT,
        TRAIN_BIN,
    )

    verify_dataset(
        tokenizer,
        VALID_TEXT,
        VALID_BIN,
    )

    print()
    print("Dataset preparation complete ✓")


if __name__ == "__main__":
    main()