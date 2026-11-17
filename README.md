# Lab 2: Parallel Word Counting in a Text Corpus with MPI

## Problem Description

The generator (`generator.py`) creates a `dataset/` folder with:

- `consulta.txt`: 200 query words, one per line.
- `file_0001.txt` ... `file_3000.txt`: the corpus, 44,951,458 words (268 MB). File sizes are intentionally uneven: about 60% are small (2k–5k words), 30% medium (10k–20k) and 10% large (50k–100k).

The task is to count how many times each query word appears in the corpus and report the 10 most frequent ones. The work is split into three parts: a sequential baseline, a first MPI version with broadcast and static file distribution, and a second MPI version that reduces the load imbalance of the first one.

## Environment and Execution Instructions

**Hardware and software used for all measurements**

| Item | Value |
| --- | --- |
| Machine | Apple M5 laptop, 10 cores (4 performance + 6 efficiency) |
| OS | macOS 27.2.1 |
| Docker | 29.5.2 |
| Container | `augustosalazar/slim-mpi:2` (Python 3.8.20, mpi4py 4.1.1, Open MPI 4.1.4) |
| Processes `p` | **8**, the value recommended by `detectar_procesos_mpi.py` (see `experiments/calibration_detectar_procesos.log`) |

**Repository files**

| File | Description |
| --- | --- |
| `generator.py` | Dataset generator (as provided, no changes) |
| `baseline_secuencial.py` | Sequential baseline (as provided, no changes) |
| `run_all.sh` | Batch script for all experiments (as provided, no changes) |
| `mpi1.py` | MPI version 1: broadcast + static block distribution |
| `mpi2.py` | MPI version 2: dynamic distribution, largest files first |
| `detectar_procesos_mpi.py` | Script from the previous lab used to choose `p` |
| `experiments/` | Raw logs of every run used in this report |

All commands are run from the repository root.

```bash
# 1. Generate the dataset
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py

# 2. Choose p
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py

# 3. Sequential baseline (also writes dataset/baseline_results.csv)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# 4. MPI version 1 and version 2
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n 8 python /app/mpi1.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n 8 python /app/mpi2.py

# 5. Optional: provided batch script (p = 2, 4, 8, three runs each)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

On Windows (Command Prompt) replace `"$(pwd)"` with `"%cd%"`.

### a. Experimental Plan

1. Generate the dataset once (fixed seed 42, so every run uses the same corpus).
2. Run `detectar_procesos_mpi.py` to choose `p`. Result: `p = 8`.
3. Run the sequential baseline 3 times and record `T_seq` and its top 10.
4. Run `mpi1.py` and `mpi2.py` 3 times each with `p = 8`. The runs are interleaved (baseline, mpi1, mpi2, baseline, ...) so that changes in machine load affect all three programs equally.
5. For each run, record the total execution time and the local time of every process. Compute `T_p` (mean of the 3 totals), `S_p = T_seq / T_p`, `E_p = S_p / p`, `LI = max(t_i) / mean(t_i)` and the relative spread `(max(t_i) − min(t_i)) / mean(t_i)`.
6. Check that every MPI run produces exactly the same counts as the baseline.
7. As extra evidence, run the provided `run_all.sh` (p = 2, 4, 8).

### b. Sequential baseline (what it does and how)

`baseline_secuencial.py` reads `consulta.txt` into a set. It then goes through every `file_*.txt` in `dataset/`, splits each line into words, lowercases them and increments a `Counter` for each word that is in the query set. It prints the elapsed time (measured with `time.perf_counter()` around the counting), the totals and the top 10, and saves all counts to `dataset/baseline_results.csv`.

### c. MPI version 1 (what it does and how)

`mpi1.py` follows the 7 required steps:

1. Rank 0 reads `consulta.txt`.
2. Rank 0 sends the query words to all processes with `comm.bcast`.
3. Rank 0 gets the sorted list of `file_*.txt` files.
4. The list is cut into `p` contiguous blocks of 375 files each and sent with `comm.scatter` (static distribution).
5. Each process counts the query words in its own files. The counting loop is the same as the baseline (`split()`, `lower()`, set lookup), so the comparison is fair. The process measures this part as its **local time**.
6. Each process sends its partial `Counter`, number of files, bytes, tokens and local time to rank 0 with `comm.gather`.
7. Rank 0 merges the counters, prints a per-process table (files, MB, tokens, local time), `LI`, the relative spread and the top 10.

The **total execution time** is measured on rank 0 with `MPI.Wtime()`, from just after a `Barrier` until the global result is built. It is printed as `EXECUTION_TIME=...`, the line that `run_all.sh` extracts.

### d. Test procedure

- **Correctness:** both MPI programs load `dataset/baseline_results.csv` and compare all 200 counts with their own result. They print `Baseline check: MATCH` only if every count is identical. All 6 measured runs and all 18 runs of `run_all.sh` printed `MATCH`, and the top 10 always matched the baseline:

  ```
  a: 785774, para: 392156, sus: 228913, otros: 105530, ante: 99832,
  unos: 88794, otra: 83901, vosotros: 61617, mios: 58420, tuya: 56635
  ```

- **Timing:** each program prints machine-readable lines (`EXECUTION_TIME=`, `LOCAL_TIMES=`, `LI=`). The values in this report come directly from the logs in `experiments/logs/`.

## Experimental Plan Execution

### a. Sequential baseline timing

| Run | Time (s) |
| ---: | ---: |
| 1 | 4.456 |
| 2 | 4.906 |
| 3 | 4.783 |
| **Mean `T_seq`** | **4.715** |

Output: 3000 files, 44,951,458 tokens, 3,631,778 occurrences, and the top 10 listed above.

### b. MPI version 1 timing results (p = 8)

| Run | Total time (s) | LI | Relative spread |
| ---: | ---: | ---: | ---: |
| 1 | 1.471 | 1.092 | 0.202 |
| 2 | 1.532 | 1.112 | 0.267 |
| 3 | 1.514 | 1.137 | 0.252 |
| **Mean** | **`T_p` = 1.506** | **1.114** | **0.240** |

`S_8 = 4.715 / 1.506 = ` **3.13**, `E_8 = 3.13 / 8 = ` **0.39**

### c. Load imbalance evidence (per-process local times and LI)

Local time `t_i` (s) of each process in version 1:

| Rank | Files | MB (run 1) | Run 1 | Run 2 | Run 3 | Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 375 | 36.73 | 1.152 | 1.219 | 1.255 | 1.209 |
| 1 | 375 | **29.37** | 1.005 | 0.988 | 1.004 | **0.999** |
| 2 | 375 | **37.82** | 1.228 | 1.300 | 1.189 | **1.239** |
| 3 | 375 | 36.57 | 1.231 | 1.257 | 1.141 | 1.210 |
| 4 | 375 | 32.32 | 1.113 | 1.060 | 1.094 | 1.089 |
| 5 | 375 | 33.86 | 1.111 | 1.098 | 1.041 | 1.083 |
| 6 | 375 | 33.10 | 1.004 | 1.147 | 1.056 | 1.069 |
| 7 | 375 | 35.28 | 1.170 | 1.281 | 1.289 | 1.247 |

Every process gets the same number of files, but not the same amount of text. Because large files are 10–50 times bigger than small ones, the bytes per block range from 29.37 MB (rank 1) to 37.82 MB (rank 2), a ratio of 1.29. The local times follow the same pattern: rank 1, with the least data, is among the fastest in every run (~1.00 s), and the ranks with more MB (0, 2, 3, 7) are the slowest (~1.21–1.25 s). The data imbalance alone (max MB / mean MB = 37.82 / 34.38) is 1.10, very close to the measured `LI = 1.11`. Most of the imbalance is caused by the static split by file count, and the rest is noise from the machine.

### d. MPI version 2: how it corrects the imbalance, and its results

**How it works (`mpi2.py`):**

1. Rank 0 reads and broadcasts the query (`comm.bcast`), as in version 1.
2. Rank 0 sorts the files **by size, largest first**, and groups them into tasks of 5 files.
3. Instead of a fixed split, tasks are assigned **dynamically** with point-to-point messages: each worker receives 2 tasks to start (so it always has the next one ready). Every time it finishes one, it sends `TAG_DONE` and rank 0 sends it a new task (`TAG_WORK`), or `TAG_STOP` when none are left.
4. Rank 0 is not idle: between replies it checks for requests with `comm.Iprobe` and processes the **smallest** remaining tasks itself, so all `p` processes count words.
5. Partial results are collected with `comm.gather`, and the report is the same as in version 1.

A process that gets slower files, or runs on a slower core, simply takes fewer tasks. Processing large files first means only small tasks are left at the end, so all processes finish at almost the same time.

**Results (p = 8):**

| Run | Total time (s) | LI | Relative spread |
| ---: | ---: | ---: | ---: |
| 1 | 1.563 | 1.052 | 0.078 |
| 2 | 1.527 | 1.039 | 0.056 |
| 3 | 1.473 | 1.026 | 0.048 |
| **Mean** | **`T_p` = 1.521** | **1.039** | **0.061** |

Per-process mean local times: 1.224, 1.166, 1.159, 1.184, 1.181, 1.182, 1.167, 1.165 s (rank 0 to 7). The MB per worker now varies only between 34.7 and 38.8 MB. Rank 0 processes many small files (19.7 MB) because it also spends time answering the other processes.

**Comparison of both versions (p = 8):**

| Metric | Sequential | MPI v1 (static) | MPI v2 (dynamic) |
| --- | ---: | ---: | ---: |
| Mean total time (s) | 4.715 | 1.506 | 1.521 |
| Speedup `S_p` | 1.00 | 3.13 | 3.10 |
| Efficiency `E_p` | – | 0.39 | 0.39 |
| Load imbalance `LI` | – | 1.114 | **1.039** |
| Relative spread | – | 0.240 | **0.061** |
| Max local time, mean of runs (s) | – | 1.273 | 1.225 |

**Other approaches tested before choosing version 2:**

| Approach | Total time p = 8 (s) | LI | Why it was not kept |
| --- | ---: | ---: | --- |
| Static greedy by size (LPT: largest file to least-loaded process) | 1.441 | 1.088 | Equal MB per process, but core speed differences still leave LI ≈ 1.09. Logs: `experiments/logs/alt_lpt_*` (separate batch, same-batch v1 = 1.424 s). |
| Shared counter with MPI one-sided `Fetch_and_op` | ~5.3 | 1.000 | Perfect balance, but every request waits for rank 0 to enter MPI, which made it 3.5× slower. |
| Dynamic with a coordinator that does not count | ~1.47 | 1.004 | Only `p − 1` processes count, and at p = 2 it took 4.9 s (one worker). |

**`run_all.sh` results** (mean of 3 runs, `experiments/run_all_times_summary.csv`, baseline in that run: 4.370 s):

| p | MPI v1 (s) | MPI v2 (s) |
| ---: | ---: | ---: |
| 2 | 2.470 | 2.593 |
| 4 | 2.086 | 2.398 |
| 8 | 1.590 | 1.587 |

## Analysis

### a. Parallel performance

Yes. Version 1 reduced the time from 4.715 s to 1.506 s, a speedup of 3.13. However, that is far from the ideal `S_8 = 8`, with an efficiency of only 0.39. The measurements explain most of the gap:

- **Each process works slower in parallel than alone (main cause).** The sum of the 8 local times is 9.0–9.4 s per run, about twice the 4.7 s that one process needs for the whole corpus. Even with perfect balance, the best possible time would be about 9.1 / 8 ≈ 1.14 s, which already limits the speedup to about 4.1. The reasons:
  - Only 4 of the 10 cores are performance cores, so with 8 processes at least 4 run on slower efficiency cores.
  - All processes share the same memory bandwidth and caches.
  - The Docker Desktop VM reads the files from the macOS folder through a shared mount.
- **Load imbalance:** the slowest process takes on average 1.27 s, about 11% above the mean (`LI = 1.11`). The total time is set by that process.
- **Fixed sequential part:** about 0.23 s of the 1.51 s are spent outside the counting: listing and checking the 3000 files on rank 0 (~0.12 s), `bcast`/`scatter`, `gather` of 8 `Counter`s and the final merge. This part does not decrease when `p` grows.

### b. Load balance

Version 1 showed clear and repeatable imbalance:

- Rank 1 finished in ~1.00 s in every run, while ranks 2 and 7 needed ~1.24–1.25 s.
- `LI` was 1.09–1.14 and the relative spread 0.20–0.27.
- This matches the MB each block received (29.4 MB to 37.8 MB), even though every process had exactly 375 files.

Version 2 reduced the imbalance a lot: `LI` went from 1.114 to 1.039, and the spread from 0.240 to 0.061 (4 times smaller). However, it **did not produce a real performance improvement**. The mean total time was 1.521 s against 1.506 s, a difference of 1%, which is smaller than the run-to-run variation (~±4%).

The reason is that the imbalance in version 1 only cost about 0.13 s (max 1.27 s against mean 1.14 s), and the dynamic scheme spends a similar amount on:

- the messages for 600 tasks,
- rank 0 dividing its time between counting and answering requests,
- sorting the files by size,
- slightly slower counting (mean local time 1.18 s against 1.14 s).

The `run_all.sh` results confirm this: at p = 8 both versions are equal (1.59 s), and at p = 2 and 4 version 2 is slower, because rank 0 counts fewer files while it serves the others. Version 2 fixes the balance, but on this dataset balance was not the main bottleneck.

### c. Limitations

- **Single, shared machine.** Everything ran on one laptop inside the Docker Desktop VM, not on a cluster. Other programs on the laptop also affect the results: the baseline varied from 4.46 s to 4.91 s (±5%) across runs. The 1% difference between v1 and v2 is inside this noise, so the data supports "no improvement" but cannot say which version is faster.
- **Heterogeneous cores.** The M5's mix of performance and efficiency cores, together with `--oversubscribe` and `--use-hwthread-cpus`, means the 8 processes do not have identical hardware. This favors dynamic schemes and makes the results specific to this machine.
- **Few repetitions.** Only 3 runs per configuration. That is enough to see the LI difference (consistent in every run), but not to give precise confidence intervals for the total times.
- **File cache.** The dataset fits in memory and was read many times, so after the first run it was served from the cache. The results show the CPU cost of counting, not the cost of reading from disk.
- **Different timing boundaries.** The baseline times only its counting loop, while the MPI times include the distribution and the `gather`, and exclude starting Python and MPI. The comparison is still reasonable, but the speedups are slightly optimistic for the user-visible time.
- **One dataset.** A single generated corpus (seed 42). With more skewed file sizes, or on several nodes, the imbalance of the static version would be larger, and version 2 would probably show a real gain.

Overall, these limitations weaken the comparison of total times between the two MPI versions. They do not weaken the conclusions about speedup (~3.1 against an ideal of 8) or about load balance, because the LI difference was large and repeated in every run.

## Conclusions

For this problem on this machine, I would recommend the **static block distribution (version 1)**, ideally with blocks balanced by bytes instead of by number of files. With `p = 8` it reached a speedup of 3.13 and an efficiency of 0.39. The dynamic version reached 3.10 and 0.39 while lowering the load imbalance from `LI = 1.114` to `LI = 1.039`, so the extra coordination costs about as much as the balance saves. Version 1 is also simpler, uses only collective operations and was faster at `p = 2` and `p = 4`. The real limit on speedup was not the distribution strategy but the per-process slowdown when 8 processes share the laptop's cores and memory (the sum of local times doubled from 4.7 s to ~9.1 s). Load balancing alone cannot fix this. Dynamic distribution would become the better choice with larger differences in file size, slower or unpredictable nodes, or a real cluster where some processes are much slower than others.
