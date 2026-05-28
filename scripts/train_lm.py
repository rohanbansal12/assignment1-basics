from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.components import LM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


@dataclass
class TrainConfig:
    train_tokens_path: Path
    valid_tokens_path: Path
    checkpoint_path: Path
    resume_path: Path | None

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    batch_size: int
    max_iters: int
    lr_max: float
    lr_min: float
    warmup_iters: int
    cosine_cycle_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip: float | None

    eval_interval: int
    eval_iters: int
    log_interval: int
    checkpoint_interval: int

    device: str
    dtype: torch.dtype
    seed: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a small Transformer language model.")

    parser.add_argument("--train-tokens-path", type=Path, required=True)
    parser.add_argument("--valid-tokens-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("out/checkpoints/latest.pt"))
    parser.add_argument("--resume-path", type=Path, default=None)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--cosine-cycle-iters", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)

    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()
    dtype_by_name = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    return TrainConfig(
        train_tokens_path=args.train_tokens_path,
        valid_tokens_path=args.valid_tokens_path,
        checkpoint_path=args.checkpoint_path,
        resume_path=args.resume_path,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        dtype=dtype_by_name[args.dtype],
        seed=args.seed,
    )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokens(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def batch_loss(model: LM, data: np.ndarray, cfg: TrainConfig) -> torch.Tensor:
    x, y = get_batch(data, cfg.batch_size, cfg.context_length, cfg.device)
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def estimate_loss(model: LM, train_data: np.ndarray, valid_data: np.ndarray, cfg: TrainConfig) -> dict[str, float]:
    was_training = model.training
    model.eval()

    losses: dict[str, float] = {}
    for split, data in [("train", train_data), ("valid", valid_data)]:
        split_losses = []
        for _ in range(cfg.eval_iters):
            loss = batch_loss(model, data, cfg)
            split_losses.append(loss.item())
        losses[split] = float(np.mean(split_losses))

    if was_training:
        model.train()
    return losses


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_data = load_tokens(cfg.train_tokens_path)
    valid_data = load_tokens(cfg.valid_tokens_path)

    model = LM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    start_iter = 0
    if cfg.resume_path is not None:
        start_iter = load_checkpoint(cfg.resume_path, model, optimizer)

    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={cfg.device} dtype={cfg.dtype}")
    print(f"parameters={count_parameters(model):,}")
    print(f"train_tokens={len(train_data):,} valid_tokens={len(valid_data):,}")

    last_log_time = time.perf_counter()
    last_log_iter = start_iter

    for iteration in range(start_iter, cfg.max_iters):
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=cfg.lr_max,
            min_learning_rate=cfg.lr_min,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.cosine_cycle_iters,
        )
        set_learning_rate(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, train_data, cfg)
        loss.backward()

        if cfg.grad_clip is not None:
            gradient_clipping(model.parameters(), cfg.grad_clip)

        optimizer.step()

        step = iteration + 1
        if step % cfg.log_interval == 0:
            now = time.perf_counter()
            elapsed = now - last_log_time
            tokens = (step - last_log_iter) * cfg.batch_size * cfg.context_length
            tokens_per_sec = tokens / elapsed if elapsed > 0 else float("nan")
            print(
                f"iter={step} loss={loss.item():.4f} lr={lr:.3e} "
                f"tokens/s={tokens_per_sec:,.0f}"
            )
            last_log_time = now
            last_log_iter = step

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, valid_data, cfg)
            valid_ppl = math.exp(losses["valid"]) if losses["valid"] < 20 else float("inf")
            print(
                f"eval iter={step} train_loss={losses['train']:.4f} "
                f"valid_loss={losses['valid']:.4f} valid_ppl={valid_ppl:.2f}"
            )

        if step % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, cfg.checkpoint_path)
            print(f"saved checkpoint to {cfg.checkpoint_path}")

    save_checkpoint(model, optimizer, cfg.max_iters, cfg.checkpoint_path)
    print(f"saved final checkpoint to {cfg.checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())
