from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from cs336_basics import bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile BPE training phases.")
    parser.add_argument("--input-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=bpe.NUM_PROCESSES)
    parser.add_argument("--profile-merges", action="store_true")
    args = parser.parse_args()

    print(
        f"file={args.input_path} size={args.input_path.stat().st_size / 1024**3:.2f} GiB "
        f"vocab_size={args.vocab_size} processes={args.num_processes}"
    )

    t0 = time.perf_counter()
    vocab = bpe._init_vocab(args.special_token)
    t1 = time.perf_counter()

    pretoken_counts = bpe._pretokenize_file_parallel(
        args.input_path,
        args.special_token,
        args.num_processes,
    )
    t2 = time.perf_counter()

    print(f"init_seconds={t1 - t0:.3f}")
    print(f"pretokenize_seconds={t2 - t1:.3f}")
    print(f"unique_pretokens={len(pretoken_counts):,}")
    print(f"total_pretokens={sum(pretoken_counts.values()):,}")
    print(
        "top_pretokens="
        + repr([(b''.join(k).decode('utf-8', errors='replace'), v) for k, v in pretoken_counts.most_common(10)])
    )

    if args.profile_merges:
        profiler = cProfile.Profile()
        profiler.enable()
        t3 = time.perf_counter()
        merges = bpe._train_merges(pretoken_counts, vocab, args.vocab_size)
        t4 = time.perf_counter()
        profiler.disable()

        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(40)
        print(stream.getvalue())
    else:
        t3 = time.perf_counter()
        merges = bpe._train_merges(pretoken_counts, vocab, args.vocab_size)
        t4 = time.perf_counter()

    print(f"merge_seconds={t4 - t3:.3f}")
    print(f"total_seconds={t4 - t0:.3f}")
    print(f"merges={len(merges):,} vocab={len(vocab):,}")


if __name__ == "__main__":
    main()
