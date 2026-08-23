from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"

OUTPUT_DIR = Path(
    "data/raw_smoltalk"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train.txt"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "test.txt"
)


# ============================================================
# Chat markers
# ============================================================

USER_MARKER = "<|USER|>"
ASSISTANT_MARKER = "<|ASSISTANT|>"
SYSTEM_MARKER = "<|SYSTEM|>"
END_OF_TURN = "<|END_OF_TURN|>"
END_OF_DIALOG = "<|END_OF_DIALOG|>"


# ============================================================
# Statistics
# ============================================================

class Stats:

    def __init__(
        self,
        split,
    ):

        self.split = split

        self.rows = 0
        self.accepted = 0
        self.rejected = 0

        self.messages = 0

        self.user_messages = 0
        self.assistant_messages = 0
        self.system_messages = 0

        self.characters = 0

        self.source_counts = {}


# ============================================================
# Structural validation
# ============================================================

def validate_messages(
    messages,
):

    if not isinstance(
        messages,
        list,
    ):
        return False

    if len(messages) == 0:
        return False

    for message in messages:

        if not isinstance(
            message,
            dict,
        ):
            return False

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if role not in (
            "user",
            "assistant",
            "system",
        ):
            return False

        if not isinstance(
            content,
            str,
        ):
            return False

        if not content.strip():
            return False

    return True


# ============================================================
# Formatting
# ============================================================

def format_conversation(
    messages,
):

    parts = []

    for message in messages:

        role = message[
            "role"
        ]

        content = (
            message[
                "content"
            ]
            .strip()
        )

        if role == "user":

            marker = (
                USER_MARKER
            )

        elif role == "assistant":

            marker = (
                ASSISTANT_MARKER
            )

        elif role == "system":

            marker = (
                SYSTEM_MARKER
            )

        else:

            raise ValueError(
                role
            )

        parts.append(
            marker
        )

        parts.append(
            content
        )

        parts.append(
            END_OF_TURN
        )

    parts.append(
        END_OF_DIALOG
    )

    return "\n".join(
        parts
    ) + "\n"


# ============================================================
# Split writer
# ============================================================

def process_split(
    split,
    output_path,
):

    print()
    print("=" * 60)

    print(
        f"RAW SMOLTALK: "
        f"{split.upper()}"
    )

    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    stats = Stats(
        split
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for row in tqdm(
            dataset,
            desc=split,
        ):

            stats.rows += 1

            messages = row.get(
                "messages"
            )

            source = row.get(
                "source",
                "unknown",
            )

            stats.source_counts[
                source
            ] = (
                stats
                .source_counts
                .get(
                    source,
                    0,
                )
                + 1
            )

            if not validate_messages(
                messages
            ):

                stats.rejected += 1

                continue

            text = (
                format_conversation(
                    messages
                )
            )

            output_file.write(
                text
            )

            stats.accepted += 1

            stats.messages += (
                len(messages)
            )

            stats.characters += (
                len(text)
            )

            for message in messages:

                role = message[
                    "role"
                ]

                if role == "user":

                    stats.user_messages += 1

                elif role == "assistant":

                    stats.assistant_messages += 1

                elif role == "system":

                    stats.system_messages += 1

    return stats


# ============================================================
# Report
# ============================================================

def print_stats(
    stats,
):

    print()
    print(
        f"Split              : "
        f"{stats.split}"
    )

    print(
        f"Rows               : "
        f"{stats.rows:,}"
    )

    print(
        f"Accepted           : "
        f"{stats.accepted:,}"
    )

    print(
        f"Rejected           : "
        f"{stats.rejected:,}"
    )

    print(
        f"Messages           : "
        f"{stats.messages:,}"
    )

    print(
        f"User messages      : "
        f"{stats.user_messages:,}"
    )

    print(
        f"Assistant messages : "
        f"{stats.assistant_messages:,}"
    )

    print(
        f"System messages    : "
        f"{stats.system_messages:,}"
    )

    print(
        f"Characters         : "
        f"{stats.characters:,}"
    )

    if stats.accepted:

        print(
            f"Avg msgs/dialog    : "
            f"{stats.messages / stats.accepted:.2f}"
        )

        print(
            f"Avg chars/dialog   : "
            f"{stats.characters / stats.accepted:.1f}"
        )

    print()
    print("Sources:")

    for source, count in sorted(
        stats.source_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"  {source:<35}"
            f"{count:>10,}"
        )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print(
        "MiniGPT-M4 V3 Raw SmolTalk Preparation"
    )
    print("=" * 60)

    print()
    print(
        "CONTENT FILTERING: DISABLED"
    )

    print(
        "Only structurally invalid rows "
        "will be rejected."
    )

    train_stats = process_split(
        "train",
        TRAIN_OUTPUT,
    )

    test_stats = process_split(
        "test",
        TEST_OUTPUT,
    )

    print()
    print("=" * 60)
    print("TRAIN STATISTICS")
    print("=" * 60)

    print_stats(
        train_stats
    )

    print()
    print("=" * 60)
    print("TEST STATISTICS")
    print("=" * 60)

    print_stats(
        test_stats
    )

    print()
    print("=" * 60)

    print(
        "RAW SMOLTALK PREPARATION COMPLETE ✓"
    )

    print("=" * 60)

    print()
    print(
        f"Train file: "
        f"{TRAIN_OUTPUT}"
    )

    print(
        f"Test file : "
        f"{TEST_OUTPUT}"
    )


if __name__ == "__main__":
    main()
