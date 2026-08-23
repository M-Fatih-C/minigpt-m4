from pathlib import Path
import re
import numpy as np
from datasets import load_dataset

from tokenizer.bpe import BPETokenizer


DATASET_NAME = "elricwan/dailydialog"
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")

DATA_DIR = Path("data")

SEED = 42

TRAIN_DIALOGUES = 11_118
VALID_DIALOGUES = 1_000
TEST_DIALOGUES = 1_000

USER_MARKER = "<|USER|>\n"
ASSISTANT_MARKER = "<|ASSISTANT|>\n"
END_OF_TURN = "\n<|END_OF_TURN|>\n"
END_OF_DIALOG = "\n<|END_OF_DIALOG|>\n"


def clean_utterance(text: str) -> str:
    """
    Dataset mirror'ındaki:

        Person A: Hello
        Person B: Hi

    prefixlerini kaldırır.
    """

    text = text.strip()

    text = re.sub(
        r"^Person\s+[AB]\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()

def append_tokens(
    tokenizer,
    text,
    tokens,
    masks,
    learn,
):
    ids = tokenizer.encode(text)

    tokens.extend(ids)

    mask_value = 1 if learn else 0

    masks.extend(
        [mask_value] * len(ids)
    )


def encode_conversation(
    tokenizer,
    conversation,
):
    tokens = []
    masks = []

    for turn_index, utterance in enumerate(conversation):

        utterance = clean_utterance(
            utterance
        )

        if not utterance:
            continue

        is_user = (
            turn_index % 2 == 0
        )

        if is_user:

            append_tokens(
                tokenizer,
                USER_MARKER,
                tokens,
                masks,
                learn=False,
            )

            append_tokens(
                tokenizer,
                utterance,
                tokens,
                masks,
                learn=False,
            )

            append_tokens(
                tokenizer,
                END_OF_TURN,
                tokens,
                masks,
                learn=False,
            )

        else:

            append_tokens(
                tokenizer,
                ASSISTANT_MARKER,
                tokens,
                masks,
                learn=False,
            )

            append_tokens(
                tokenizer,
                utterance,
                tokens,
                masks,
                learn=True,
            )

            append_tokens(
                tokenizer,
                END_OF_TURN,
                tokens,
                masks,
                learn=True,
            )

    # İki farklı diyaloğun birbirine yapışmasını
    # engelleyen explicit boundary.
    append_tokens(
        tokenizer,
        END_OF_DIALOG,
        tokens,
        masks,
        learn=False,
    )

    return tokens, masks


def write_split(
    tokenizer,
    dataset,
    name,
):
    token_path = (
        DATA_DIR
        / f"dialog_{name}.bin"
    )

    mask_path = (
        DATA_DIR
        / f"dialog_{name}_mask.bin"
    )

    total_tokens = 0
    learned_tokens = 0

    print()
    print("=" * 60)
    print(f"Preparing dialogue split: {name}")
    print("=" * 60)

    with (
        token_path.open("wb") as token_file,
        mask_path.open("wb") as mask_file,
    ):

        for index, row in enumerate(dataset):

            conversation = row[
                "conversation"
            ]

            tokens, masks = (
                encode_conversation(
                    tokenizer,
                    conversation,
                )
            )

            np.asarray(
                tokens,
                dtype=np.uint16,
            ).tofile(token_file)

            np.asarray(
                masks,
                dtype=np.uint8,
            ).tofile(mask_file)

            total_tokens += len(tokens)
            learned_tokens += sum(masks)

            if (
                (index + 1) % 1000
                == 0
            ):
                print(
                    f"\rDialogs: {index + 1:>6,}"
                    f" | tokens: {total_tokens:>10,}"
                    f" | assistant targets: "
                    f"{learned_tokens:>10,}",
                    end="",
                    flush=True,
                )

    print()
    print()

    print(
        f"Dialogues          : "
        f"{len(dataset):,}"
    )

    print(
        f"Total tokens       : "
        f"{total_tokens:,}"
    )

    print(
        f"Assistant targets  : "
        f"{learned_tokens:,}"
    )

    print(
        f"Target percentage  : "
        f"{100 * learned_tokens / total_tokens:.2f}%"
    )

    print(
        f"Token file         : "
        f"{token_path}"
    )

    print(
        f"Mask file          : "
        f"{mask_path}"
    )

    return token_path, mask_path


def verify_split(
    token_path,
    mask_path,
):

    tokens = np.memmap(
        token_path,
        dtype=np.uint16,
        mode="r",
    )

    masks = np.memmap(
        mask_path,
        dtype=np.uint8,
        mode="r",
    )

    assert len(tokens) == len(masks)

    assert np.all(
        (masks == 0)
        | (masks == 1)
    )

    assert masks.sum() > 0

    print(
        f"Verification passed ✓ "
        f"({token_path.name})"
    )


def show_example(
    tokenizer,
    conversation,
):
    print()
    print("=" * 60)
    print("FORMATTED EXAMPLE")
    print("=" * 60)
    print()

    for i, utterance in enumerate(
        conversation
    ):

        role = (
            "USER"
            if i % 2 == 0
            else "ASSISTANT"
        )

        print(
            f"<|{role}|>"
        )

        print(
            utterance.strip()
        )

        print(
            "<|END_OF_TURN|>"
        )

    tokens, masks = encode_conversation(
        tokenizer,
        conversation,
    )

    print()
    print(
        f"Example tokens          : "
        f"{len(tokens)}"
    )

    print(
        f"Assistant target tokens : "
        f"{sum(masks)}"
    )


def main():

    print("=" * 60)
    print("MiniGPT-M4 DailyDialog Preparation")
    print("=" * 60)

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    print(
        f"Tokenizer vocab: "
        f"{tokenizer.vocab_size:,}"
    )

    print()
    print(
        f"Loading {DATASET_NAME}..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(
        f"Total dialogues: "
        f"{len(dataset):,}"
    )

    # Aynı split'i her çalıştırmada
    # tekrar üretmek için deterministic shuffle.
    dataset = dataset.shuffle(
        seed=SEED
    )

    train_end = TRAIN_DIALOGUES

    validation_end = (
        train_end
        + VALID_DIALOGUES
    )

    test_end = (
        validation_end
        + TEST_DIALOGUES
    )

    train = dataset.select(
        range(
            0,
            train_end,
        )
    )

    validation = dataset.select(
        range(
            train_end,
            validation_end,
        )
    )

    test = dataset.select(
        range(
            validation_end,
            test_end,
        )
    )

    show_example(
        tokenizer,
        train[0]["conversation"],
    )

    train_paths = write_split(
        tokenizer,
        train,
        "train",
    )

    validation_paths = write_split(
        tokenizer,
        validation,
        "validation",
    )

    test_paths = write_split(
        tokenizer,
        test,
        "test",
    )

    print()
    print("=" * 60)
    print("VERIFY")
    print("=" * 60)

    verify_split(
        *train_paths
    )

    verify_split(
        *validation_paths
    )

    verify_split(
        *test_paths
    )

    print()
    print("=" * 60)
    print("DAILYDIALOG PREPARATION COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()