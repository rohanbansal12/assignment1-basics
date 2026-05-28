import os
from collections import Counter, defaultdict
from multiprocessing import Pool
import regex as re
from typing import BinaryIO
import time
from pathlib import Path
import pickle
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = 12


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _init_vocab(special_tokens)
    pretoken_counts = _pretokenize_file_parallel(input_path, special_tokens, NUM_PROCESSES)
    merges = _train_merges(pretoken_counts, vocab, vocab_size)

    return vocab, merges


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    seen: set[bytes] = set(vocab.values())

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in seen:
            vocab[len(vocab)] = token_bytes
            seen.add(token_bytes)

    return vocab


def _pretokenize_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:

    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="ignore")

    if special_tokens:
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(special_pattern, text)
    else:
        segments = [text]

    counts = Counter()
    for segment in segments:
        for match in re.finditer(PAT, segment):
            pretoken_str = match.group(0)
            pretoken_bytes = pretoken_str.encode("utf-8")
            pretoken_tuple = tuple(bytes([b]) for b in pretoken_bytes)
            counts[pretoken_tuple] += 1
    return counts


def _pretokenize_file_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int,
) -> Counter[tuple[bytes, ...]]:
    num_processes = max(1, num_processes)

    with open(input_path, "rb") as f:
        if special_tokens:
            chunk_boundaries = find_chunk_boundaries(
                file=f, desired_num_chunks=num_processes, split_special_token=special_tokens[0].encode("utf-8")
            )
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            chunk_boundaries = [0, file_size]

    jobs = [
        (input_path, chunk_boundaries[i], chunk_boundaries[i + 1], special_tokens)
        for i in range(len(chunk_boundaries) - 1)
    ]

    if num_processes == 1:
        total = Counter()
        for job in jobs:
            total.update(_pretokenize_chunk(*job))
    else:
        with Pool(processes=num_processes) as pool:
            worker_counters = pool.starmap(_pretokenize_chunk, jobs)

        total = Counter()
        for counter in worker_counters:
            total.update(counter)

    return total


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_pairs(
    token_counts: dict[tuple[bytes, ...], int],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for tokens, count in token_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            pair_to_pretokens[pair].add(tokens)

    return pair_counts, pair_to_pretokens


def _merge_pretoken(
    pretoken: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0

    while i < len(pretoken):
        if i < len(pretoken) - 1 and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(pretoken[i])
            i += 1

    return tuple(merged)


class _MaxPairHeapItem:
    """Heap item for selecting the best BPE pair.

    heapq is a min-heap, so __lt__ is written such that the item we want
    first is considered "smaller": higher count first, and for ties,
    lexicographically larger pair first. This preserves the old selection rule:

        max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
    """

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_MaxPairHeapItem") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _build_pair_heap(
    pair_counts: Counter[tuple[bytes, bytes]],
) -> list[_MaxPairHeapItem]:
    heap = [_MaxPairHeapItem(count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)
    return heap


def _push_pair_if_active(
    heap: list[_MaxPairHeapItem],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair: tuple[bytes, bytes],
) -> None:
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(heap, _MaxPairHeapItem(count, pair))


def _pop_best_pair(
    heap: list[_MaxPairHeapItem],
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes] | None:
    """Return the current best pair, discarding stale heap entries.

    Pair counts are mutated after every merge. Instead of updating/removing old
    heap entries in place, we push new entries and lazily skip stale ones here.
    """
    while heap:
        item = heapq.heappop(heap)
        if pair_counts.get(item.pair, 0) == item.count:
            return item.pair
    return None


def _train_merges(
    pretoken_counts: Counter[tuple[bytes, ...]],
    vocab: dict[int, bytes],
    vocab_size: int,
) -> list[tuple[bytes, bytes]]:

    merges = []

    pair_counts, pair_to_pretokens = _count_pairs(pretoken_counts)
    pair_heap = _build_pair_heap(pair_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        max_pair = _pop_best_pair(pair_heap, pair_counts)
        if max_pair is None:
            break

        affected = list(pair_to_pretokens[max_pair])

        replacement_counts = Counter()
        for token in affected:
            new_pretoken = _merge_pretoken(token, max_pair)
            replacement_counts[new_pretoken] += pretoken_counts[token]

        for token in affected:
            old_count = pretoken_counts[token]
            pretoken_counts.pop(token)

            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] -= old_count
                if pair_counts[pair] == 0:
                    pair_counts.pop(pair)
                else:
                    _push_pair_if_active(pair_heap, pair_counts, pair)
                if token in pair_to_pretokens[pair]:
                    pair_to_pretokens[pair].remove(token)
                if not pair_to_pretokens[pair]:
                    pair_to_pretokens.pop(pair)

        for token, count in replacement_counts.items():
            pretoken_counts[token] += count

            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += count
                _push_pair_if_active(pair_heap, pair_counts, pair)
                pair_to_pretokens[pair].add(token)

        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]

    return merges


def train_bpe_tinystories():
    repo_dir = Path(__file__).parent.parent

    input_path = repo_dir / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    t = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    elapsed = time.time() - t

    out_dir = repo_dir / "out"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "tinystories_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(out_dir / "tinystories_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_token = max(vocab.values(), key=len)

    print(f"Finished training BPE on tinystories: {elapsed} s")
    print(f"Longest Token: {longest_token.decode('utf-8', errors='replace')}")


def train_bpe_openweb():
    repo_dir = Path(__file__).parent.parent

    input_path = repo_dir / "data" / "owt_train.txt"
    vocab_size = 32000
    special_tokens = []

    t = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    elapsed = time.time() - t

    out_dir = repo_dir / "out"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "opentext_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(out_dir / "opentext_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_token = max(vocab.values(), key=len)

    print(f"Finished training BPE on opentext: {elapsed} s")
    print(f"Longest Token: {longest_token.decode('utf-8', errors='replace')}")


if __name__ == "__main__":
    ## train_bpe_tinystories()

    train_bpe_openweb()
