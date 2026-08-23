from array import array
from pathlib import Path
import sys
import time

from tokenizers import Tokenizer
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

TOKENIZER_PATH = Path(
    "tokenizer_v3/tokenizer.json"
)

TRAIN_TEXT_PATH = Path(
    "data/pretrain/fineweb_edu_train.txt"
)

VALIDATION_TEXT_PATH = Path(
    "data/pretrain/fineweb_edu_validation.txt"
)

OUTPUT_DIR = Path(
    "data/pretrain_v3"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train.bin"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "validation.bin"
)


# Exactly 500M training tokens.
TRAIN_TARGET_TOKENS = 500_000_000


# Batch several documents before sending them
# through the Rust tokenizer.
ENCODE_BATCH_SIZE = 256


END_OF_TEXT_MARKER = (
    "<|END_OF_TEXT|>"
)


# ============================================================
# Document iterator
# ============================================================

def iter_documents(
    path: Path,
):
    """
    FineWeb preparation format:

        document text
        <|END_OF_TEXT|>
        document text
        <|END_OF_TEXT|>

    We reconstruct documents and emit them one at a time.

    The textual marker itself is NOT encoded as ordinary
    text. We append the dedicated atomic EOT token manually.
    """

    buffer = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if (
                line.strip()
                == END_OF_TEXT_MARKER
            ):

                if buffer:

                    yield "".join(
                        buffer
                    )

                    buffer.clear()

                continue

            buffer.append(
                line
            )

    if buffer:

        yield "".join(
            buffer
        )


# ============================================================
# Binary writer
# ============================================================

class TokenWriter:

    def __init__(
        self,
        path: Path,
    ):
        self.path = path

        self.file = path.open(
            "wb"
        )

        self.count = 0

    def write(
        self,
        token_ids,
    ):

        if not token_ids:
            return

        values = array(
            "H",
            token_ids,
        )

        # Binary format is explicitly little-endian.
        if sys.byteorder != "little":
            values.byteswap()

        values.tofile(
            self.file
        )

        self.count += len(
            token_ids
        )

    def close(self):

        self.file.close()


# ============================================================
# Batch encoding
# ============================================================

def encode_batch(
    tokenizer: Tokenizer,
    documents,
    eot_id: int,
):

    encodings = (
        tokenizer.encode_batch(
            documents,
            add_special_tokens=False,
        )
    )

    output = []

    for encoding in encodings:

        ids = encoding.ids

        # Every document receives an explicit boundary.
        output.append(
            ids + [eot_id]
        )

    return output


# ============================================================
# Train
# ============================================================

def prepare_train(
    tokenizer: Tokenizer,
    eot_id: int,
):

    print()
    print("=" * 70)
    print("TRAIN TOKENIZATION")
    print("=" * 70)

    print()
    print(
        f"Target tokens : "
        f"{TRAIN_TARGET_TOKENS:,}"
    )

    writer = TokenWriter(
        TRAIN_OUTPUT
    )

    document_batch = []

    document_count = 0

    start_time = time.perf_counter()

    progress = tqdm(
        total=TRAIN_TARGET_TOKENS,
        desc="train tokens",
        unit="tok",
    )

    stop = False

    for document in iter_documents(
        TRAIN_TEXT_PATH
    ):

        document_batch.append(
            document
        )

        if (
            len(document_batch)
            < ENCODE_BATCH_SIZE
        ):
            continue

        token_batches = encode_batch(
            tokenizer,
            document_batch,
            eot_id,
        )

        document_batch.clear()

        for token_ids in token_batches:

            remaining = (
                TRAIN_TARGET_TOKENS
                - writer.count
            )

            if remaining <= 0:

                stop = True
                break

            # Entire document fits.
            if len(token_ids) <= remaining:

                writer.write(
                    token_ids
                )

                progress.update(
                    len(token_ids)
                )

            else:
                # Final document must be truncated because
                # the training binary is exactly 500M tokens.
                #
                # Preserve the EOT boundary as final token.
                if remaining == 1:

                    final_tokens = [
                        eot_id
                    ]

                else:

                    final_tokens = (
                        token_ids[
                            :remaining - 1
                        ]
                        + [eot_id]
                    )

                writer.write(
                    final_tokens
                )

                progress.update(
                    len(final_tokens)
                )

                stop = True
                break

            document_count += 1

        if stop:
            break

    # Corpus ended before target. Normally this should
    # never happen for the current FineWeb sample.
    if (
        not stop
        and document_batch
    ):

        token_batches = encode_batch(
            tokenizer,
            document_batch,
            eot_id,
        )

        for token_ids in token_batches:

            remaining = (
                TRAIN_TARGET_TOKENS
                - writer.count
            )

            if remaining <= 0:
                break

            if len(token_ids) <= remaining:

                writer.write(
                    token_ids
                )

                progress.update(
                    len(token_ids)
                )

            else:

                if remaining == 1:

                    final_tokens = [
                        eot_id
                    ]

                else:

                    final_tokens = (
                        token_ids[
                            :remaining - 1
                        ]
                        + [eot_id]
                    )

                writer.write(
                    final_tokens
                )

                progress.update(
                    len(final_tokens)
                )

                break

            document_count += 1

    progress.close()

    writer.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if (
        writer.count
        != TRAIN_TARGET_TOKENS
    ):

        raise RuntimeError(
            "FineWeb corpus ended before "
            "500M tokens were produced. "
            f"Got {writer.count:,} tokens."
        )

    print()
    print(
        f"Documents      : "
        f"{document_count:,}+"
    )

    print(
        f"Tokens         : "
        f"{writer.count:,}"
    )

    print(
        f"Elapsed        : "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Throughput     : "
        f"{writer.count / elapsed:,.0f} tok/s"
    )

    return writer.count


# ============================================================
# Validation
# ============================================================

def prepare_validation(
    tokenizer: Tokenizer,
    eot_id: int,
):

    print()
    print("=" * 70)
    print("VALIDATION TOKENIZATION")
    print("=" * 70)

    writer = TokenWriter(
        VALIDATION_OUTPUT
    )

    document_batch = []

    document_count = 0

    start_time = time.perf_counter()

    for document in tqdm(
        iter_documents(
            VALIDATION_TEXT_PATH
        ),
        desc="validation docs",
        unit="doc",
    ):

        document_batch.append(
            document
        )

        if (
            len(document_batch)
            < ENCODE_BATCH_SIZE
        ):
            continue

        token_batches = encode_batch(
            tokenizer,
            document_batch,
            eot_id,
        )

        for token_ids in token_batches:

            writer.write(
                token_ids
            )

            document_count += 1

        document_batch.clear()

    if document_batch:

        token_batches = encode_batch(
            tokenizer,
            document_batch,
            eot_id,
        )

        for token_ids in token_batches:

            writer.write(
                token_ids
            )

            document_count += 1

    writer.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Documents      : "
        f"{document_count:,}"
    )

    print(
        f"Tokens         : "
        f"{writer.count:,}"
    )

    print(
        f"Elapsed        : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Throughput     : "
        f"{writer.count / elapsed:,.0f} tok/s"
    )

    return writer.count


# ============================================================
# Verification
# ============================================================

def verify_binary(
    path: Path,
    expected_tokens=None,
):

    file_bytes = (
        path.stat().st_size
    )

    if file_bytes % 2 != 0:

        raise RuntimeError(
            f"Invalid uint16 file size: "
            f"{path}"
        )

    token_count = (
        file_bytes // 2
    )

    if (
        expected_tokens is not None
        and token_count
        != expected_tokens
    ):

        raise RuntimeError(
            f"Token count mismatch: "
            f"{token_count:,} "
            f"!= "
            f"{expected_tokens:,}"
        )

    return token_count


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "MiniGPT-M4 V3 "
        "FineWeb Token Preparation"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = (
        Tokenizer.from_file(
            str(
                TOKENIZER_PATH
            )
        )
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    print()
    print(
        f"Vocabulary     : "
        f"{vocab_size:,}"
    )

    if vocab_size != 8192:

        raise RuntimeError(
            "Expected 8192-token vocabulary."
        )

    eot_id = (
        tokenizer.token_to_id(
            END_OF_TEXT_MARKER
        )
    )

    if eot_id is None:

        raise RuntimeError(
            "Missing EOT token."
        )

    print(
        f"EOT token ID   : "
        f"{eot_id}"
    )

    if eot_id != 2:

        raise RuntimeError(
            f"Expected EOT ID 2, got {eot_id}."
        )

    train_tokens = (
        prepare_train(
            tokenizer,
            eot_id,
        )
    )

    validation_tokens = (
        prepare_validation(
            tokenizer,
            eot_id,
        )
    )

    verified_train = (
        verify_binary(
            TRAIN_OUTPUT,
            TRAIN_TARGET_TOKENS,
        )
    )

    verified_validation = (
        verify_binary(
            VALIDATION_OUTPUT
        )
    )

    if (
        verified_train
        != train_tokens
    ):

        raise RuntimeError(
            "Train verification failed."
        )

    if (
        verified_validation
        != validation_tokens
    ):

        raise RuntimeError(
            "Validation verification failed."
        )

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print()
    print(
        f"Train tokens      : "
        f"{train_tokens:,}"
    )

    print(
        f"Validation tokens : "
        f"{validation_tokens:,}"
    )

    print()
    print(
        f"Train binary      : "
        f"{TRAIN_OUTPUT}"
    )

    print(
        f"Validation binary : "
        f"{VALIDATION_OUTPUT}"
    )

    print()
    print("=" * 70)
    print(
        "PRETRAIN DATASET READY ✓"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
