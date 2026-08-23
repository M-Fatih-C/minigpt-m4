import random

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


# ============================================================
# MiniGPT-M4 V3 Tokenizer
# ============================================================

VOCAB_SIZE = 8192

SEED = 42


# ------------------------------------------------------------
# Source corpora
# ------------------------------------------------------------

FINEWEB_PATH = Path(
    "data/pretrain/fineweb_edu_train.txt"
)

SMOLTALK_PATH = Path(
    "data/raw_smoltalk/train.txt"
)


# ------------------------------------------------------------
# Tokenizer training corpus
# ------------------------------------------------------------

CORPUS_DIR = Path(
    "data/tokenizer_v3"
)

CORPUS_PATH = (
    CORPUS_DIR
    / "training_corpus.txt"
)

TOKENIZER_DIR = Path(
    "tokenizer_v3"
)

TOKENIZER_PATH = (
    TOKENIZER_DIR
    / "tokenizer.json"
)


# ------------------------------------------------------------
# Corpus mixture
#
# 75% general language
# 25% instruction / conversation
# ------------------------------------------------------------

FINEWEB_SAMPLE_BYTES = (
    192 * 1024**2
)

SMOLTALK_SAMPLE_BYTES = (
    64 * 1024**2
)

SAMPLE_CHUNK_BYTES = (
    1024 * 1024
)


# ------------------------------------------------------------
# Compression benchmark
# ------------------------------------------------------------

BENCHMARK_BYTES = (
    8 * 1024**2
)


# ------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------

SPECIAL_TOKENS = [
    "<|PAD|>",
    "<|UNK|>",
    "<|END_OF_TEXT|>",
    "<|SYSTEM|>",
    "<|USER|>",
    "<|ASSISTANT|>",
    "<|END_OF_TURN|>",
    "<|END_OF_DIALOG|>",
]


# ============================================================
# Helpers
# ============================================================

def mib(
    value: int,
) -> float:

    return (
        value
        / 1024**2
    )


def random_text_chunk(
    path: Path,
    byte_count: int,
    rng: random.Random,
) -> bytes:

    file_size = (
        path.stat().st_size
    )

    if file_size <= byte_count:

        with path.open(
            "rb"
        ) as file:

            return file.read()

    max_offset = (
        file_size
        - byte_count
        - 1
    )

    offset = rng.randint(
        0,
        max_offset,
    )

    with path.open(
        "rb"
    ) as file:

        file.seek(
            offset
        )

        # If we started inside a line,
        # discard the partial first line.
        if offset > 0:

            file.readline()

        data = file.read(
            byte_count
        )

    # Avoid ending in the middle of a line.
    last_newline = (
        data.rfind(b"\n")
    )

    if last_newline > 0:

        data = data[
            :last_newline + 1
        ]

    return data


def sample_source(
    source_path: Path,
    target_bytes: int,
    output_file,
    rng: random.Random,
    label: str,
) -> int:

    print()
    print(
        f"Sampling {label}..."
    )

    print(
        f"Source : {source_path}"
    )

    print(
        f"Target : "
        f"{mib(target_bytes):.0f} MiB"
    )

    written = 0

    while written < target_bytes:

        remaining = (
            target_bytes
            - written
        )

        request_size = min(
            SAMPLE_CHUNK_BYTES,
            remaining,
        )

        chunk = random_text_chunk(
            source_path,
            request_size,
            rng,
        )

        if not chunk:
            continue

        text = chunk.decode(
            "utf-8",
            errors="ignore",
        )

        encoded = text.encode(
            "utf-8"
        )

        output_file.write(
            text
        )

        written += len(
            encoded
        )

        percent = (
            written
            / target_bytes
            * 100
        )

        print(
            f"\r"
            f"{label:<12}"
            f"{mib(written):>8.1f} MiB"
            f" / "
            f"{mib(target_bytes):.0f} MiB"
            f" "
            f"({percent:>6.2f}%)",
            end="",
            flush=True,
        )

    print()

    return written


# ============================================================
# Build tokenizer corpus
# ============================================================

def build_training_corpus():

    rng = random.Random(
        SEED
    )

    CORPUS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "BUILD TOKENIZER TRAINING CORPUS"
    )
    print("=" * 70)

    with CORPUS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        fineweb_bytes = sample_source(
            FINEWEB_PATH,
            FINEWEB_SAMPLE_BYTES,
            output_file,
            rng,
            "FineWeb",
        )

        smoltalk_bytes = sample_source(
            SMOLTALK_PATH,
            SMOLTALK_SAMPLE_BYTES,
            output_file,
            rng,
            "SmolTalk",
        )

    total = (
        fineweb_bytes
        + smoltalk_bytes
    )

    print()
    print(
        f"FineWeb sample : "
        f"{mib(fineweb_bytes):.2f} MiB"
    )

    print(
        f"SmolTalk sample: "
        f"{mib(smoltalk_bytes):.2f} MiB"
    )

    print(
        f"Total corpus   : "
        f"{mib(total):.2f} MiB"
    )

    print(
        f"Corpus file    : "
        f"{CORPUS_PATH}"
    )


# ============================================================
# Train tokenizer
# ============================================================

def train_tokenizer():

    print()
    print("=" * 70)
    print(
        "TRAIN BYTE-LEVEL BPE TOKENIZER"
    )
    print("=" * 70)

    print()
    print(
        f"Target vocabulary : "
        f"{VOCAB_SIZE:,}"
    )

    print(
        f"Special tokens    : "
        f"{len(SPECIAL_TOKENS)}"
    )

    tokenizer = Tokenizer(
        BPE(
            unk_token="<|UNK|>"
        )
    )

    # Byte-level tokenizer.
    tokenizer.pre_tokenizer = (
        ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
    )

    tokenizer.decoder = (
        ByteLevelDecoder()
    )

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=(
            ByteLevel.alphabet()
        ),
    )

    tokenizer.train(
        files=[
            str(
                CORPUS_PATH
            )
        ],
        trainer=trainer,
    )

    actual_vocab_size = (
        tokenizer.get_vocab_size()
    )

    print()
    print(
        f"Actual vocabulary : "
        f"{actual_vocab_size:,}"
    )

    if actual_vocab_size != VOCAB_SIZE:

        raise RuntimeError(
            f"Tokenizer vocab mismatch: "
            f"{actual_vocab_size} "
            f"!= "
            f"{VOCAB_SIZE}"
        )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(
            TOKENIZER_PATH
        )
    )

    print(
        f"Tokenizer saved   : "
        f"{TOKENIZER_PATH}"
    )

    return tokenizer


# ============================================================
# Tests
# ============================================================

def test_special_tokens(
    tokenizer: Tokenizer,
):

    print()
    print("=" * 70)
    print(
        "SPECIAL TOKEN TEST"
    )
    print("=" * 70)

    for special_token in (
        SPECIAL_TOKENS
    ):

        token_id = (
            tokenizer.token_to_id(
                special_token
            )
        )

        encoded = (
            tokenizer.encode(
                special_token,
                add_special_tokens=False,
            )
        )

        print(
            f"{special_token:<20}"
            f"id={token_id:<6}"
            f"encoded={encoded.ids}"
        )

        if token_id is None:

            raise RuntimeError(
                f"Missing special token: "
                f"{special_token}"
            )

        if (
            len(encoded.ids) != 1
            or encoded.ids[0] != token_id
        ):

            raise RuntimeError(
                f"Special token does not "
                f"encode as one token: "
                f"{special_token}"
            )

    print()
    print(
        "All special tokens are atomic ✓"
    )


def roundtrip_tests(
    tokenizer: Tokenizer,
):

    print()
    print("=" * 70)
    print(
        "ROUNDTRIP TEST"
    )
    print("=" * 70)

    examples = [
        (
            "Hello world! "
            "This is MiniGPT-M4."
        ),
        (
            "The transformer uses "
            "rotary positional embeddings."
        ),
        (
            "def hello(name):\n"
            "    return f\"Hello {name}!\""
        ),
        (
            "Merhaba dünya! "
            "Türkçe karakterler: "
            "çğıöşü ÇĞİÖŞÜ."
        ),
        (
            "2 + 2 = 4, "
            "but 123456789 is also text."
        ),
    ]

    for index, text in enumerate(
        examples,
        start=1,
    ):

        encoded = (
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

        decoded = (
            tokenizer.decode(
                encoded.ids,
                skip_special_tokens=False,
            )
        )

        print()
        print(
            f"Example {index}"
        )

        print(
            f"Text   : "
            f"{repr(text)}"
        )

        print(
            f"Tokens : "
            f"{len(encoded.ids)}"
        )

        print(
            f"Decoded: "
            f"{repr(decoded)}"
        )

        if decoded != text:

            raise RuntimeError(
                "Tokenizer roundtrip failed."
            )

    print()
    print(
        "Roundtrip tests passed ✓"
    )


# ============================================================
# Compression benchmark
# ============================================================

def compression_test(
    tokenizer: Tokenizer,
    path: Path,
    label: str,
    rng: random.Random,
):

    raw = random_text_chunk(
        path,
        BENCHMARK_BYTES,
        rng,
    )

    text = raw.decode(
        "utf-8",
        errors="ignore",
    )

    byte_count = len(
        text.encode(
            "utf-8"
        )
    )

    encoded = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    token_count = len(
        encoded.ids
    )

    bytes_per_token = (
        byte_count
        / token_count
    )

    print(
        f"{label:<12}"
        f"bytes="
        f"{byte_count:>10,}"
        f" | tokens="
        f"{token_count:>10,}"
        f" | bytes/token="
        f"{bytes_per_token:.3f}"
    )

    decoded = tokenizer.decode(
        encoded.ids,
        skip_special_tokens=False,
    )

    if decoded != text:

        raise RuntimeError(
            f"{label} compression "
            f"roundtrip failed."
        )


def compression_benchmark(
    tokenizer: Tokenizer,
):

    print()
    print("=" * 70)
    print(
        "COMPRESSION BENCHMARK"
    )
    print("=" * 70)

    print()

    rng = random.Random(
        SEED + 100
    )

    compression_test(
        tokenizer,
        FINEWEB_PATH,
        "FineWeb",
        rng,
    )

    compression_test(
        tokenizer,
        SMOLTALK_PATH,
        "SmolTalk",
        rng,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "MiniGPT-M4 V3 — 8K Byte-Level BPE"
    )
    print("=" * 70)

    print()
    print(
        "Vocabulary : "
        f"{VOCAB_SIZE:,}"
    )

    print(
        "FineWeb mix: "
        "192 MiB"
    )

    print(
        "SmolTalk mix: "
        "64 MiB"
    )

    build_training_corpus()

    tokenizer = (
        train_tokenizer()
    )

    test_special_tokens(
        tokenizer
    )

    roundtrip_tests(
        tokenizer
    )

    compression_benchmark(
        tokenizer
    )

    print()
    print("=" * 70)
    print(
        "MINIGPT-M4 V3 TOKENIZER READY ✓"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
