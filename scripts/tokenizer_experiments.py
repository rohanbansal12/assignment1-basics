from pathlib import Path
from cs336_basics.tokenizer import Tokenizer
import time

if __name__ == "__main__":
    out_dir = Path("__file__").parent.parent / "out"
    tokenizer = Tokenizer.from_files(out_dir / "tinystories_bpe_vocab.pkl", out_dir / "tinystories_bpe_merges.pkl", ["<|endoftext|>"])

    input_path = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    file_size_bytes = input_path.stat().st_size

    start = time.perf_counter()

    num_tokens = 0
    with open(input_path, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            num_tokens += 1

    elapsed = time.perf_counter() - start

    bytes_per_second = file_size_bytes / elapsed
    tokens_per_second = num_tokens / elapsed
    bytes_per_token = file_size_bytes / num_tokens

    print(f"Tokenizer stats: {bytes_per_second:2f} bytes/s, {tokens_per_second:2f} tokens/s, {bytes_per_token:2f} bytes/token")
    
    pile_bytes = 825 * 1024**3
    pile_seconds = pile_bytes / bytes_per_second
    pile_hours = pile_seconds / 3600

    print(f"Tokenizer Pile Estimate: {pile_hours} hr")