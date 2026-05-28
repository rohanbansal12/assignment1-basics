import torch
from collections.abc import Iterable, Callable
import numpy as np
import numpy.typing as npt
import os
import typing


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps"

    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=(batch_size, 1))
    offsets = np.arange(context_length)

    x_np = dataset[starts + offsets]
    y_np = dataset[starts + offsets + 1]

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    return x, y


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_z = shifted.exp().sum(dim=-1).log()
    loss = log_z - shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + np.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * np.pi)
        ) * (max_learning_rate - min_learning_rate)

    else:
        return min_learning_rate


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    torch.nn.utils.clip_grad_norm_(params, max_l2_norm, norm_type=2.0, foreach=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {}
    obj["iteration"] = iteration
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                state["m"].lerp_(grad, 1 - betas[0])
                state["v"].lerp_(grad.pow(2), 1 - betas[1])

                m_hat = state["m"] / (1 - betas[0] ** t)
                v_hat = state["v"] / (1 - betas[1] ** t)

                with torch.no_grad():
                    p.add_(m_hat / (v_hat.sqrt() + eps) + weight_decay * p, alpha=-lr)

        return loss
