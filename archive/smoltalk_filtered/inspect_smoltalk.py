import random

import numpy as np

from smoltalk_dataset import SmolTalkDataset
from tokenizer.bpe import BPETokenizer


TOKENIZER_PATH = "tokenizer/tokenizer.json"

IGNORE_INDEX = -100

SEED = 42
SAMPLE_COUNT = 10


def main():

    random.seed(SEED)

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    dataset = SmolTalkDataset(
        "train"
    )

    print()
    print("=" * 80)
    print("SMOLTALK TRAINING EXAMPLE INSPECTION")
    print("=" * 80)

    example_ids = random.sample(
        range(dataset.example_count),
        SAMPLE_COUNT,
    )

    for number, example_id in enumerate(
        example_ids,
        start=1,
    ):

        tokens, masks = dataset.get_example(
            example_id
        )

        print()
        print("=" * 80)
        print(
            f"EXAMPLE {number} | ID {example_id}"
        )
        print("=" * 80)

        full_text = tokenizer.decode(
            tokens.tolist()
        )

        print()
        print("FULL EXAMPLE:")
        print("-" * 80)
        print(full_text)

        # --------------------------------------------------
        # Context / target ayrımı
        # --------------------------------------------------

        context_tokens = []
        target_tokens = []

        for token, mask in zip(
            tokens,
            masks,
        ):

            if mask == 1:
                target_tokens.append(
                    int(token)
                )
            else:
                context_tokens.append(
                    int(token)
                )

        print()
        print("SUPERVISED TARGET:")
        print("-" * 80)

        print(
            tokenizer.decode(
                target_tokens
            )
        )

        print()
        print(
            f"Total tokens  : {len(tokens)}"
        )

        print(
            f"Target tokens : {int(masks.sum())}"
        )

        print(
            f"Target ratio  : "
            f"{100 * masks.mean():.2f}%"
        )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()