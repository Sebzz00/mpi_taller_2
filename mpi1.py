import os
from collections import Counter

from mpi4py import MPI

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
QUERY_FILE = "consulta.txt"
BASELINE_CSV = "baseline_results.csv"
TOP_N = 10


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def list_corpus_files(dataset_dir):
    return sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith("file_") and name.endswith(".txt")
        and os.path.isfile(os.path.join(dataset_dir, name))
    )


def count_words(files, query):
    counts = Counter()
    tokens = 0
    for name in files:
        with open(os.path.join(DATASET_DIR, name), "r", encoding="utf-8") as f:
            for line in f:
                words = [w.lower() for w in line.split()]
                tokens += len(words)
                for w in words:
                    if w in query:
                        counts[w] += 1
    return counts, tokens


def block_split(files, size):
    n = len(files)
    return [files[i * n // size:(i + 1) * n // size] for i in range(size)]


def load_baseline(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        return {word: int(count) for word, count in (line.strip().split(",") for line in f)}


def report(title, size, results, total_counts, total_time):
    local_times = [r["time"] for r in results]
    mean_time = sum(local_times) / len(local_times)
    li = max(local_times) / mean_time
    spread = (max(local_times) - min(local_times)) / mean_time

    print(f"=== {title} ===")
    print(f"Processes: {size} (counting ranks listed: {len(results)})\n")
    print(f"{'Rank':>4} {'Files':>6} {'MB':>8} {'Tokens':>10} {'Local time (s)':>15}")
    for r in results:
        print(f"{r['rank']:>4} {r['files']:>6} {r['bytes'] / 1e6:>8.2f} "
              f"{r['tokens']:>10} {r['time']:>15.4f}")

    print(f"\nMean local time: {mean_time:.4f} s")
    print(f"Max local time: {max(local_times):.4f} s")
    print(f"Load imbalance LI = max/mean: {li:.4f}")
    print(f"Relative spread (max-min)/mean: {spread:.4f}\n")

    print(f"Files processed: {sum(r['files'] for r in results)}")
    print(f"Total tokens read: {sum(r['tokens'] for r in results)}")
    print(f"Total occurrences found: {sum(total_counts.values())}\n")

    print(f"Top {TOP_N} query words in the corpus:")
    for word, count in total_counts.most_common(TOP_N):
        print(f"  {word}: {count}")

    baseline = load_baseline(os.path.join(DATASET_DIR, BASELINE_CSV))
    if baseline is None:
        print("\nBaseline check: baseline_results.csv not found")
    else:
        same = baseline == {w: c for w, c in total_counts.items() if c > 0}
        print(f"\nBaseline check: {'MATCH' if same else 'MISMATCH'}")

    print(f"\nLOCAL_TIMES={','.join(f'{t:.6f}' for t in local_times)}")
    print(f"LI={li:.6f}")
    print(f"EXECUTION_TIME={total_time:.6f}")


def run(title, split_files):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    t0 = MPI.Wtime()

    if rank == 0:
        if not os.path.isdir(DATASET_DIR):
            print(f"Error: dataset folder not found at {DATASET_DIR}")
            comm.Abort(1)
        query = load_query(os.path.join(DATASET_DIR, QUERY_FILE))
        files = list_corpus_files(DATASET_DIR)
        chunks = split_files(files, size)
    else:
        query = None
        chunks = None

    query = comm.bcast(query, root=0)
    my_files = comm.scatter(chunks, root=0)

    t_start = MPI.Wtime()
    counts, tokens = count_words(my_files, query)
    local_time = MPI.Wtime() - t_start

    partial = {
        "rank": rank,
        "files": len(my_files),
        "bytes": sum(os.path.getsize(os.path.join(DATASET_DIR, f)) for f in my_files),
        "tokens": tokens,
        "time": local_time,
        "counts": counts,
    }
    results = comm.gather(partial, root=0)

    if rank == 0:
        total_counts = Counter()
        for r in results:
            total_counts.update(r["counts"])
        total_time = MPI.Wtime() - t0
        report(title, size, results, total_counts, total_time)


if __name__ == "__main__":
    run("MPI version 1: static block distribution", block_split)
