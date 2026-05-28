from dataclasses import dataclass
import torch
from cs336_basics.components import LM, softmax
from cs336_basics.tokenizer import Tokenizer


@dataclass
class DecodeConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float | None = None
    eos_token_id: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "mps"


def decode(
    model: LM,
    tokenizer: Tokenizer,
    prompt: str,
    config: DecodeConfig,
) -> str:
    """
    Generate text from `model` starting from `prompt`.
    """
    model.eval()
    ids = tokenizer.encode(prompt)

    for _ in range(config.max_new_tokens):
        context = ids[-model.context_length:]
        x = torch.tensor(context, device=config.device)[None, :]  # batch size 1

        with torch.no_grad():
            logits = model(x)

        next_logits = logits[0, -1, :]
        next_id = sample_next_token(next_logits, config.temperature, config.top_p)

        ids.append(next_id)

        if config.eos_token_id is not None and next_id == config.eos_token_id:
            break

    return tokenizer.decode(ids)

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float | None,
) -> int:
    """
    Convert next-token logits into one sampled token ID.
    """
    if temperature <= 0:
        return torch.argmax(logits).item()
    
    probs = softmax(logits / temperature, dim=-1)
    if top_p is not None:
        probs = apply_top_p(probs, top_p)

    return torch.multinomial(probs, num_samples=1).item()

def apply_top_p(
    probs: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """
    Keep the smallest high-probability token set whose cumulative probability
    reaches top_p, zero the rest, then renormalize.
    """
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # Mask tokens after we pass top_p.
    remove_mask_sorted = cumulative_probs > top_p
    remove_mask_sorted[1:] = remove_mask_sorted[:-1].clone()
    remove_mask_sorted[0] = False

    sorted_probs = sorted_probs.masked_fill(remove_mask_sorted, 0.0)

    sorted_probs = sorted_probs / sorted_probs.sum()

    filtered_probs = torch.zeros_like(probs)
    filtered_probs[sorted_indices] = sorted_probs

    return filtered_probs