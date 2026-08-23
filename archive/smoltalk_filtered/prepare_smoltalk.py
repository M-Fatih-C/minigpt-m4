from pathlib import Path

import numpy as np
from datasets import load_dataset

from config import BLOCK_SIZE
from tokenizer.bpe import BPETokenizer


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
SOURCE_NAME = "smol-magpie-ultra-short"

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
DATA_DIR = Path("data")

SEED = 42

# Her 10 conversation'dan 1 tanesi validation'a gider.
# Split conversation seviyesinde yapılır.
VALIDATION_EVERY = 10

# Model BLOCK_SIZE=256.
#
# Eğitimde:
#
# tokens[:-1] -> X
# tokens[1:]  -> Y
#
# kullandığımız için bir örnek maksimum 257 token olabilir.
MAX_SEQUENCE_TOKENS = BLOCK_SIZE + 1


# ============================================================
# Chat format
# ============================================================

USER_MARKER = "<|USER|>\n"
ASSISTANT_MARKER = "<|ASSISTANT|>\n"
END_OF_TURN = "\n<|END_OF_TURN|>\n"


# ============================================================
# Helpers
# ============================================================

def encode_text(
    tokenizer: BPETokenizer,
    text: str,
) -> list[int]:
    return tokenizer.encode(
        text.strip()
    )


def build_chunk(
    tokenizer: BPETokenizer,
    role: str,
    content: str,
    learn: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Bir USER veya ASSISTANT turn'ünü tokenlara çevirir.

    learn=False:
        Bu tokenlar context olarak kullanılır,
        loss hesaplanmaz.

    learn=True:
        Bu tokenlar supervised target olur.
    """

    if role == "user":
        marker = USER_MARKER

    elif role == "assistant":
        marker = ASSISTANT_MARKER

    else:
        raise ValueError(
            f"Unknown role: {role}"
        )

    tokens: list[int] = []
    masks: list[int] = []

    # --------------------------------------------------------
    # Role marker
    # --------------------------------------------------------

    marker_tokens = tokenizer.encode(
        marker
    )

    tokens.extend(
        marker_tokens
    )

    # Role marker üzerinde loss hesaplamıyoruz.
    masks.extend(
        [0] * len(marker_tokens)
    )

    # --------------------------------------------------------
    # Message content
    # --------------------------------------------------------

    content_tokens = encode_text(
        tokenizer,
        content,
    )

    tokens.extend(
        content_tokens
    )

    masks.extend(
        [1 if learn else 0]
        * len(content_tokens)
    )

    # --------------------------------------------------------
    # End of turn
    # --------------------------------------------------------

    eot_tokens = tokenizer.encode(
        END_OF_TURN
    )

    tokens.extend(
        eot_tokens
    )

    # Assistant cevabının nerede biteceğini
    # modelin öğrenmesini istiyoruz.
    masks.extend(
        [1 if learn else 0]
        * len(eot_tokens)
    )

    return tokens, masks


# ============================================================
# Conversation validation
# ============================================================

def valid_conversation(
    messages,
) -> bool:
    """
    Yalnızca küçük modelimize uygun,
    temiz user/assistant konuşmalarını kabul eder.
    """

    if not messages:
        return False

    # En az:
    #
    # USER
    # ASSISTANT
    #
    # olmalı.
    if len(messages) < 2:
        return False

    # Küçük model için çok uzun diyalogları almıyoruz.
    if len(messages) > 8:
        return False

    # System/tool gibi roller şimdilik yok.
    for message in messages:

        role = message.get(
            "role"
        )

        if role not in (
            "user",
            "assistant",
        ):
            return False

        content = message.get(
            "content"
        )

        if not isinstance(
            content,
            str,
        ):
            return False

        if not content.strip():
            return False

    # Conversation USER ile başlamalı.
    if (
        messages[0]["role"]
        != "user"
    ):
        return False

    # Roller düzgün sıra izlemeli:
    #
    # user
    # assistant
    # user
    # assistant
    # ...
    expected_role = "user"

    for message in messages:

        if (
            message["role"]
            != expected_role
        ):
            return False

        expected_role = (
            "assistant"
            if expected_role == "user"
            else "user"
        )

    return True


# ============================================================
# Training-example builder
# ============================================================

def build_training_examples(
    tokenizer: BPETokenizer,
    messages,
) -> list[
    tuple[
        list[int],
        list[int],
    ]
]:
    """
    Her assistant cevabı için AYRI bir supervised training
    example oluşturur.

    Örnek conversation:

        USER:
        My name is Fatih.

        ASSISTANT:
        Nice to meet you, Fatih.

        USER:
        What is my name?

        ASSISTANT:
        Your name is Fatih.

    Buradan iki örnek çıkar:

    Example 1:
        USER  -> context
        ASSISTANT -> target

    Example 2:
        önceki USER      -> context
        önceki ASSISTANT -> context
        mevcut USER      -> context
        mevcut ASSISTANT -> target

    Loss sadece SON assistant cevabında hesaplanır.
    """

    examples = []

    # Tamamlanmış conversation turnlerini tutar.
    #
    # [
    #   ("user", tokens),
    #   ("assistant", tokens),
    #   ...
    # ]
    chunks = []

    for message in messages:

        role = message["role"]

        content = (
            message["content"]
            .strip()
        )

        # ----------------------------------------------------
        # USER
        # ----------------------------------------------------

        if role == "user":

            user_tokens, _ = (
                build_chunk(
                    tokenizer,
                    role="user",
                    content=content,
                    learn=False,
                )
            )

            chunks.append(
                (
                    "user",
                    user_tokens,
                )
            )

            continue

        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        assistant_tokens, assistant_masks = (
            build_chunk(
                tokenizer,
                role="assistant",
                content=content,
                learn=True,
            )
        )

        # Assistant'tan hemen önce USER bulunmalı.
        if not chunks:
            continue

        if (
            chunks[-1][0]
            != "user"
        ):
            continue

        # ----------------------------------------------------
        # Current USER + current ASSISTANT
        # ----------------------------------------------------

        current_user_tokens = (
            chunks[-1][1]
        )

        selected = [
            (
                current_user_tokens,
                [0]
                * len(
                    current_user_tokens
                ),
            ),
            (
                assistant_tokens,
                assistant_masks,
            ),
        ]

        total_length = (
            len(current_user_tokens)
            + len(assistant_tokens)
        )

        # En son user/assistant çifti bile context'e
        # sığmıyorsa bu assistant targetını kullanamayız.
        if (
            total_length
            > MAX_SEQUENCE_TOKENS
        ):

            # Yine de sonraki mesajların context'i
            # açısından assistant'ı conversation'a ekle.
            full_assistant_tokens = (
                tokenizer.encode(
                    ASSISTANT_MARKER
                    + content
                    + END_OF_TURN
                )
            )

            chunks.append(
                (
                    "assistant",
                    full_assistant_tokens,
                )
            )

            continue

        # ----------------------------------------------------
        # Önceki dialogue pair'lerini context'e geri ekle
        # ----------------------------------------------------

        # chunks şu anda örneğin:
        #
        # user1
        # assistant1
        # user2   <- current
        #
        # Dolayısıyla önceki assistant:
        # len(chunks)-2
        previous_index = (
            len(chunks) - 2
        )

        while (
            previous_index >= 1
        ):

            previous_assistant = (
                chunks[
                    previous_index
                ]
            )

            previous_user = (
                chunks[
                    previous_index - 1
                ]
            )

            if (
                previous_user[0]
                != "user"
                or
                previous_assistant[0]
                != "assistant"
            ):
                break

            pair_length = (
                len(previous_user[1])
                + len(
                    previous_assistant[1]
                )
            )

            # Context limitini aşma.
            if (
                total_length
                + pair_length
                > MAX_SEQUENCE_TOKENS
            ):
                break

            # Önceki USER
            selected.insert(
                0,
                (
                    previous_user[1],
                    [0]
                    * len(
                        previous_user[1]
                    ),
                ),
            )

            # Önceki ASSISTANT
            selected.insert(
                1,
                (
                    previous_assistant[1],
                    [0]
                    * len(
                        previous_assistant[1]
                    ),
                ),
            )

            total_length += (
                pair_length
            )

            previous_index -= 2

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        tokens: list[int] = []
        masks: list[int] = []

        for (
            part_tokens,
            part_masks,
        ) in selected:

            tokens.extend(
                part_tokens
            )

            masks.extend(
                part_masks
            )

        # Çok anlamsız/minik assistant cevaplarını
        # eğitimden çıkar.
        if sum(masks) < 4:
            continue

        assert (
            len(tokens)
            == len(masks)
        )

        assert (
            len(tokens)
            <= MAX_SEQUENCE_TOKENS
        )

        examples.append(
            (
                tokens,
                masks,
            )
        )

        # ----------------------------------------------------
        # Assistant turnünü sonraki dialogue turnleri için
        # context'e ekle.
        # ----------------------------------------------------

        full_assistant_tokens = (
            tokenizer.encode(
                ASSISTANT_MARKER
                + content
                + END_OF_TURN
            )
        )

        chunks.append(
            (
                "assistant",
                full_assistant_tokens,
            )
        )

    return examples


# ============================================================
# Writer
# ============================================================

class ExampleWriter:

    def __init__(
        self,
        name: str,
    ):

        self.name = name

        self.token_path = (
            DATA_DIR
            / f"smoltalk_{name}_tokens.bin"
        )

        self.mask_path = (
            DATA_DIR
            / f"smoltalk_{name}_mask.bin"
        )

        self.offset_path = (
            DATA_DIR
            / f"smoltalk_{name}_offsets.npy"
        )

        self.token_file = (
            self.token_path.open(
                "wb"
            )
        )

        self.mask_file = (
            self.mask_path.open(
                "wb"
            )
        )

        # offsets:
        #
        # [0, example1_end, example2_end, ...]
        self.offsets = [0]

        self.example_count = 0
        self.total_tokens = 0
        self.target_tokens = 0

    def write(
        self,
        tokens: list[int],
        masks: list[int],
    ) -> None:

        if (
            len(tokens)
            != len(masks)
        ):
            raise ValueError(
                "Token/mask length mismatch."
            )

        if (
            len(tokens)
            > MAX_SEQUENCE_TOKENS
        ):
            raise ValueError(
                "Example exceeds context limit."
            )

        np.asarray(
            tokens,
            dtype=np.uint16,
        ).tofile(
            self.token_file
        )

        np.asarray(
            masks,
            dtype=np.uint8,
        ).tofile(
            self.mask_file
        )

        self.total_tokens += (
            len(tokens)
        )

        self.target_tokens += (
            sum(masks)
        )

        self.offsets.append(
            self.total_tokens
        )

        self.example_count += 1

    def close(
        self,
    ) -> None:

        self.token_file.close()
        self.mask_file.close()

        np.save(
            self.offset_path,
            np.asarray(
                self.offsets,
                dtype=np.int64,
            ),
        )

        if (
            self.example_count > 0
        ):

            average_length = (
                self.total_tokens
                / self.example_count
            )

            target_percentage = (
                100
                * self.target_tokens
                / self.total_tokens
            )

        else:

            average_length = 0.0
            target_percentage = 0.0

        print()
        print(
            f"Examples       : "
            f"{self.example_count:,}"
        )

        print(
            f"Tokens         : "
            f"{self.total_tokens:,}"
        )

        print(
            f"Target tokens  : "
            f"{self.target_tokens:,}"
        )

        print(
            f"Target ratio   : "
            f"{target_percentage:.2f}%"
        )

        print(
            f"Avg length     : "
            f"{average_length:.1f}"
        )


# ============================================================
# Verification
# ============================================================

def verify_split(
    name: str,
) -> None:

    prefix = (
        DATA_DIR
        / f"smoltalk_{name}"
    )

    tokens = np.memmap(
        f"{prefix}_tokens.bin",
        dtype=np.uint16,
        mode="r",
    )

    masks = np.memmap(
        f"{prefix}_mask.bin",
        dtype=np.uint8,
        mode="r",
    )

    offsets = np.load(
        f"{prefix}_offsets.npy",
        mmap_mode="r",
    )

    if (
        len(tokens)
        != len(masks)
    ):
        raise RuntimeError(
            f"{name}: token/mask length mismatch."
        )

    if len(offsets) < 2:
        raise RuntimeError(
            f"{name}: no examples found."
        )

    if offsets[0] != 0:
        raise RuntimeError(
            f"{name}: first offset must be 0."
        )

    if (
        offsets[-1]
        != len(tokens)
    ):
        raise RuntimeError(
            f"{name}: final offset mismatch."
        )

    if not np.all(
        offsets[1:]
        >= offsets[:-1]
    ):
        raise RuntimeError(
            f"{name}: offsets are not sorted."
        )

    if not np.all(
        (masks == 0)
        | (masks == 1)
    ):
        raise RuntimeError(
            f"{name}: invalid mask value."
        )

    if masks.sum() == 0:
        raise RuntimeError(
            f"{name}: no supervised targets."
        )

    example_count = (
        len(offsets) - 1
    )

    max_length = int(
        np.diff(offsets).max()
    )

    min_length = int(
        np.diff(offsets).min()
    )

    print(
        f"{name}: verification passed ✓"
    )

    print(
        f"  Examples   : "
        f"{example_count:,}"
    )

    print(
        f"  Min length : "
        f"{min_length}"
    )

    print(
        f"  Max length : "
        f"{max_length}"
    )


# ============================================================
# Main
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("MiniGPT-M4 SmolTalk Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    print(
        f"Tokenizer vocab : "
        f"{tokenizer.vocab_size:,}"
    )

    print(
        f"Source          : "
        f"{SOURCE_NAME}"
    )

    print(
        f"Max sequence    : "
        f"{MAX_SEQUENCE_TOKENS}"
    )

    print(
        f"Validation      : "
        f"1/{VALIDATION_EVERY} conversations"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading streaming dataset..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Streaming shuffle:
    #
    # tam random shuffle değildir ama bizim validation
    # split'imiz için yeterli ve deterministic.
    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=10_000,
    )

    # --------------------------------------------------------
    # Writers
    # --------------------------------------------------------

    train_writer = ExampleWriter(
        "train"
    )

    validation_writer = (
        ExampleWriter(
            "validation"
        )
    )

    accepted_conversations = 0
    rejected_conversations = 0

    source_rows = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for row in dataset:

        # Sadece istediğimiz SmolTalk source.
        if (
            row.get("source")
            != SOURCE_NAME
        ):
            continue

        source_rows += 1

        messages = row.get(
            "messages"
        )

        if not valid_conversation(
            messages
        ):

            rejected_conversations += 1

            continue

        examples = (
            build_training_examples(
                tokenizer,
                messages,
            )
        )

        if not examples:

            rejected_conversations += 1

            continue

        accepted_conversations += 1

        # ----------------------------------------------------
        # Conversation-level deterministic 90/10 split
        # ----------------------------------------------------

        if (
            accepted_conversations
            % VALIDATION_EVERY
            == 0
        ):

            writer = (
                validation_writer
            )

        else:

            writer = train_writer

        # Conversation'dan çıkan bütün examples
        # AYNI split'e gider.
        for (
            tokens,
            masks,
        ) in examples:

            writer.write(
                tokens,
                masks,
            )

        if (
            accepted_conversations
            % 1000
            == 0
        ):

            print(
                f"\rConversations: "
                f"{accepted_conversations:,}"
                f" | train examples: "
                f"{train_writer.example_count:,}"
                f" | val examples: "
                f"{validation_writer.example_count:,}",
                end="",
                flush=True,
            )

    print()
    print()

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print(
        f"Source rows             : "
        f"{source_rows:,}"
    )

    print(
        f"Accepted conversations  : "
        f"{accepted_conversations:,}"
    )

    print(
        f"Rejected conversations  : "
        f"{rejected_conversations:,}"
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAIN")
    print("=" * 60)

    train_writer.close()

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    validation_writer.close()

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VERIFY")
    print("=" * 60)
    print()

    verify_split(
        "train"
    )

    print()

    verify_split(
        "validation"
    )

    print()
    print("=" * 60)
    print(
        "SMOLTALK PREPARATION COMPLETE ✓"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()