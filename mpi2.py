import os
from collections import Counter

from mpi4py import MPI

from mpi1 import DATASET_DIR, QUERY_FILE, count_words, list_corpus_files, load_query, report

TAG_WORK = 1
TAG_DONE = 2
TAG_STOP = 3
CHUNK = 5
PREFETCH = 2


def file_size(name):
    return os.path.getsize(os.path.join(DATASET_DIR, name))


def partial_result(rank, files, tokens, busy, counts):
    return {"rank": rank, "files": len(files), "bytes": sum(file_size(f) for f in files),
            "tokens": tokens, "time": busy, "counts": counts}


def coordinator(comm, files, query):
    size = comm.Get_size()
    tasks = [files[i:i + CHUNK] for i in range(0, len(files), CHUNK)]
    pending = [0] * size
    stopped = [False] * size

    for _ in range(PREFETCH):
        for w in range(1, size):
            if tasks:
                comm.send(tasks.pop(0), dest=w, tag=TAG_WORK)
                pending[w] += 1

    counts, tokens, busy, mine = Counter(), 0, 0.0, []

    def serve(w):
        pending[w] -= 1
        if tasks:
            comm.send(tasks.pop(0), dest=w, tag=TAG_WORK)
            pending[w] += 1
        elif not stopped[w]:
            comm.send(None, dest=w, tag=TAG_STOP)
            stopped[w] = True

    while tasks:
        status = MPI.Status()
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status):
            comm.recv(source=status.Get_source(), tag=TAG_DONE)
            serve(status.Get_source())
        if tasks:
            task = tasks.pop()
            t = MPI.Wtime()
            c, n = count_words(task, query)
            busy += MPI.Wtime() - t
            counts.update(c)
            tokens += n
            mine.extend(task)

    for w in range(1, size):
        if not stopped[w]:
            comm.send(None, dest=w, tag=TAG_STOP)
            stopped[w] = True
    while sum(pending):
        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status)
        pending[status.Get_source()] -= 1

    return partial_result(0, mine, tokens, busy, counts)


def worker(comm, query):
    counts, tokens, busy, mine = Counter(), 0, 0.0, []
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            break
        t = MPI.Wtime()
        c, n = count_words(task, query)
        busy += MPI.Wtime() - t
        counts.update(c)
        tokens += n
        mine.extend(task)
        comm.send(None, dest=0, tag=TAG_DONE)
    return partial_result(comm.Get_rank(), mine, tokens, busy, counts)


def main():
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
        files = sorted(list_corpus_files(DATASET_DIR), key=file_size, reverse=True)
    else:
        query = None
        files = None
    query = comm.bcast(query, root=0)

    partial = coordinator(comm, files, query) if rank == 0 else worker(comm, query)
    results = comm.gather(partial, root=0)

    if rank == 0:
        total_counts = Counter()
        for r in results:
            total_counts.update(r["counts"])
        report("MPI version 2: dynamic distribution, largest files first", size, results, total_counts, MPI.Wtime() - t0)


if __name__ == "__main__":
    main()
