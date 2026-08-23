import argparse
import json
import os
import subprocess
import sys
import time


MODELS = [
    {
        "name": "15M",
        "embd": 384,
        "heads": 6,
        "layers": 8,
    },
    {
        "name": "30M",
        "embd": 448,
        "heads": 7,
        "layers": 12,
    },
    {
        "name": "50M",
        "embd": 576,
        "heads": 9,
        "layers": 12,
    },
    {
        "name": "80M",
        "embd": 640,
        "heads": 10,
        "layers": 16,
    },
]


def gb(value):
    if value is None:
        return None

    return value / (1024 ** 3)


def worker(args):
    import torch

    from config import (
        BLOCK_SIZE,
        HEAD_SIZE,
        N_EMBD,
        N_HEAD,
        N_LAYER,
        VOCAB_SIZE,
    )

    from model.gpt import MiniGPT

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available."
        )

    device = torch.device("mps")

    torch.manual_seed(42)

    torch.mps.empty_cache()

    model = MiniGPT().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            args.batch,
            BLOCK_SIZE,
        ),
        dtype=torch.long,
        device=device,
    )

    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            args.batch,
            BLOCK_SIZE,
        ),
        dtype=torch.long,
        device=device,
    )

    def train_step():

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            x,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        return loss.item()

    # ----------------------------------------------
    # Warmup
    # ----------------------------------------------

    for _ in range(args.warmup):
        train_step()

    torch.mps.synchronize()

    # ----------------------------------------------
    # Benchmark
    # ----------------------------------------------

    start = time.perf_counter()

    losses = []

    for _ in range(args.steps):
        losses.append(
            train_step()
        )

    torch.mps.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    total_tokens = (
        args.batch
        * BLOCK_SIZE
        * args.steps
    )

    tokens_per_second = (
        total_tokens
        / elapsed
    )

    step_ms = (
        elapsed
        / args.steps
        * 1000
    )

    allocated = None
    driver = None
    recommended = None

    if hasattr(
        torch.mps,
        "current_allocated_memory",
    ):
        allocated = (
            torch.mps
            .current_allocated_memory()
        )

    if hasattr(
        torch.mps,
        "driver_allocated_memory",
    ):
        driver = (
            torch.mps
            .driver_allocated_memory()
        )

    if hasattr(
        torch.mps,
        "recommended_max_memory",
    ):
        recommended = (
            torch.mps
            .recommended_max_memory()
        )

    result = {
        "ok": True,

        "parameters":
            parameter_count,

        "parameters_m":
            parameter_count / 1e6,

        "embedding":
            N_EMBD,

        "heads":
            N_HEAD,

        "head_size":
            HEAD_SIZE,

        "layers":
            N_LAYER,

        "block":
            BLOCK_SIZE,

        "batch":
            args.batch,

        "step_ms":
            step_ms,

        "tokens_per_second":
            tokens_per_second,

        "loss":
            sum(losses)
            / len(losses),

        "allocated_gb":
            gb(allocated),

        "driver_gb":
            gb(driver),

        "recommended_gb":
            gb(recommended),
    }

    print(
        "__RESULT__"
        + json.dumps(result)
    )


def run_parent(args):

    print("=" * 88)
    print("MiniGPT-M4 Scale Benchmark")
    print("=" * 88)

    print(
        f"Context : 256"
    )

    print(
        f"Batch   : {args.batch}"
    )

    print(
        f"Warmup  : {args.warmup}"
    )

    print(
        f"Steps   : {args.steps}"
    )

    print()

    results = []

    for spec in MODELS:

        print(
            f"Benchmarking "
            f"{spec['name']}..."
        )

        env = os.environ.copy()

        env[
            "MINIGPT_N_EMBD"
        ] = str(
            spec["embd"]
        )

        env[
            "MINIGPT_N_HEAD"
        ] = str(
            spec["heads"]
        )

        env[
            "MINIGPT_N_LAYER"
        ] = str(
            spec["layers"]
        )

        env[
            "MINIGPT_BLOCK_SIZE"
        ] = "256"

        command = [
            sys.executable,
            __file__,
            "--worker",
            "--batch",
            str(args.batch),
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
        ]

        process = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:

            print(
                f"  FAILED"
            )

            print(
                process.stderr[-1500:]
            )

            results.append(
                {
                    "name":
                        spec["name"],
                    "ok":
                        False,
                }
            )

            continue

        result_line = None

        for line in (
            process.stdout
            .splitlines()
        ):

            if line.startswith(
                "__RESULT__"
            ):

                result_line = (
                    line[
                        len(
                            "__RESULT__"
                        ):
                    ]
                )

        if result_line is None:

            print(
                "  No benchmark "
                "result returned."
            )

            continue

        result = json.loads(
            result_line
        )

        result["name"] = (
            spec["name"]
        )

        results.append(
            result
        )

        print(
            f"  "
            f"{result['parameters_m']:.2f}M"
            f" | "
            f"{result['step_ms']:.1f} ms/step"
            f" | "
            f"{result['tokens_per_second']:,.0f} tok/s"
        )

    print()
    print("=" * 88)
    print("RESULTS")
    print("=" * 88)

    header = (
        f"{'Model':<8}"
        f"{'Params':>10}"
        f"{'Batch':>8}"
        f"{'ms/step':>12}"
        f"{'tok/s':>14}"
        f"{'MPS GB':>12}"
        f"{'Driver GB':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        if not result.get(
            "ok",
            False,
        ):

            print(
                f"{result['name']:<8}"
                f"{'FAILED':>10}"
            )

            continue

        allocated = (
            result[
                "allocated_gb"
            ]
        )

        driver = (
            result[
                "driver_gb"
            ]
        )

        allocated_text = (
            f"{allocated:.2f}"
            if allocated is not None
            else "-"
        )

        driver_text = (
            f"{driver:.2f}"
            if driver is not None
            else "-"
        )

        print(
            f"{result['name']:<8}"
            f"{result['parameters_m']:>9.2f}M"
            f"{result['batch']:>8}"
            f"{result['step_ms']:>12.1f}"
            f"{result['tokens_per_second']:>14,.0f}"
            f"{allocated_text:>12}"
            f"{driver_text:>12}"
        )

    print()

    valid = [
        r
        for r in results
        if r.get(
            "ok",
            False,
        )
    ]

    if valid:

        print(
            "Benchmark complete ✓"
        )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    if args.worker:
        worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
