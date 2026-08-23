import torch

from model.gpt import MiniGPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT = "chat_checkpoints/chat_step_001500.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

USER_MARKER = "<|USER|>\n"
ASSISTANT_MARKER = "<|ASSISTANT|>\n"
END_OF_TURN = "\n<|END_OF_TURN|>\n"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = MiniGPT()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded chat checkpoint: "
        f"{checkpoint['step']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']}"
    )

    return model


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    history,
    device,
):
    prompt = ""

    for role, text in history:

        if role == "user":

            prompt += (
                USER_MARKER
                + text.strip()
                + END_OF_TURN
            )

        else:

            prompt += (
                ASSISTANT_MARKER
                + text.strip()
                + END_OF_TURN
            )

    # Yeni cevabın başlangıcı.
    prompt += ASSISTANT_MARKER

    prompt_tokens = tokenizer.encode(
        prompt
    )

    # Model context'i 256 token.
    #
    # En son konuşmayı koruyoruz.
    if len(prompt_tokens) > 240:
        prompt_tokens = prompt_tokens[-240:]

    x = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        x,
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
    )

    new_tokens = generated[
        0,
        len(prompt_tokens):
    ]

    answer = tokenizer.decode(
        new_tokens
        .cpu()
        .tolist()
    )

    # Assistant cevabı bittiyse sonraki
    # konuşmayı göstermiyoruz.
    if "<|END_OF_TURN|>" in answer:

        answer = answer.split(
            "<|END_OF_TURN|>",
            1,
        )[0]

    # Model yeni role geçmeye kalkarsa kes.
    if "<|USER|>" in answer:

        answer = answer.split(
            "<|USER|>",
            1,
        )[0]

    if "<|ASSISTANT|>" in answer:

        answer = answer.split(
            "<|ASSISTANT|>",
            1,
        )[0]

    return answer.strip()


def main():

    device = get_device()

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    model = load_model(
        device
    )

    history = []

    print()
    print("=" * 60)
    print("MiniGPT-M4 Chat")
    print("=" * 60)

    print(
        "Commands: /exit, /clear"
    )

    print()

    while True:

        user_input = input(
            "You: "
        ).strip()

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input == "/clear":

            history.clear()

            print(
                "Conversation cleared.\n"
            )

            continue

        history.append(
            (
                "user",
                user_input,
            )
        )

        answer = generate_answer(
            model,
            tokenizer,
            history,
            device,
        )

        print(
            f"MiniGPT: {answer}"
        )

        print()

        history.append(
            (
                "assistant",
                answer,
            )
        )


if __name__ == "__main__":
    main()