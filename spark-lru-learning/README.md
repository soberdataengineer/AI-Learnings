# LRU caching for a Spark embedding job

## Real embeddings with Ollama

The Spark CLI now defaults to real embeddings using `nomic-embed-text`.
Start Ollama in a separate terminal (`ollama serve`), then install the model:

```bash
ollama pull nomic-embed-text
cd /Users/mohitpradhan/AI/spark-lru-learning
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_PYTHON=/usr/bin/python3 spark-submit --master 'local[2]' --conf spark.ui.enabled=false spark_job.py --provider ollama
```

`embedding_service.py` calls Ollama's `/api/embed` endpoint on cache misses.
The cache key remains `(model, text)`; hits reuse the vector without a request.
This is an exact-text cache, not semantic similarity matching. Vectors are kept
in partition-local memory; this job does not persist them or perform retrieval.

Use `--embedding-model` to select another installed embedding model and
`--ollama-url` to select the server. Every worker must reach that URL:
`127.0.0.1` works for this local example, but points at each worker's own machine
in a cluster. Pin the model version for a run; changing a model behind its tag
mid-run can mix embeddings. Comparison allows small floating-point differences.
No automatic retries are added, and API errors fail the task visibly.

For the original dependency-free simulator, add `--provider simulated` to the
Spark command. `demo.py` and the partition function's default remain simulated
so the original exercises and unit tests run without Ollama.

Run `python3 -m unittest -v` for eleven tests, including mocked Ollama request,
cache reuse, and malformed response checks. Live Ollama/Spark validation must
be run separately; a mocked response does not verify model inference.

The milestones below describe the original simulated baseline; their Spark
commands require `--provider simulated` to reproduce those results.

## Problem

An online shopping company processes support tickets to find relevant help articles. Generating an embedding for repeated ticket text wastes computation. Build a cache that retains recently used embeddings without changing the output for any ticket. This milestone implements embedding enrichment; searching a help-article database is a later extension.

- Key: `(embedding_model, text)`.
- Value: embedding vector.
- Capacity: start with 2 entries.
- Cache lookup, update, and eviction: average O(1).
- Implementation: dictionary mapping keys to nodes, plus a doubly linked list.

## Milestone 1: understand and verify the cache

No dependencies, Spark installation, or model downloads are needed yet:

```bash
cd /Users/mohitpradhan/AI/spark-lru-learning
python3 demo.py
python3 -m unittest -v
```

The simulation produces stable toy vectors. They do not represent semantic meaning and are not real AI embeddings.

Read `lru_cache.py` first. The dictionary locates nodes; the list tracks eviction order:

```text
nodes[key] -> Node(key, value, prev, next)

HEAD <-> most recent <-> ... <-> least recent <-> TAIL
```

`head` and `tail` are dummy nodes and consume no cache capacity. `get()` moves the same node to the front; `put()` updates an existing node or inserts a new one. If capacity is exceeded, the node before `tail` is removed from both structures. A missing key raises `KeyError` so that `None` remains a valid cached value. This cache assumes sequential access.

The demo processes tickets in this order:

| Ticket | Text | Result | Order: most to least recent |
|---|---|---|---|
| 101 | Reset my password | Miss | Reset |
| 102 | Payment failed | Miss | Payment, Reset |
| 103 | Reset my password | Hit | Reset, Payment |
| 104 | Account locked | Miss | Locked, Reset |
| 105 | Payment failed | Miss | Payment, Locked |

Expected totals: 5 tickets, 4 service calls, 1 cache hit. The tests also verify that cached and uncached computations produce the same vectors.

## Milestone 2: run the Spark job

Spark 4.0.0 and Java 17 are already installed on this machine. No additional packages are needed when using the existing `spark-submit` command.

```bash
cd /Users/mohitpradhan/AI/spark-lru-learning
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_PYTHON=/usr/bin/python3 spark-submit --master 'local[2]' --conf spark.ui.enabled=false spark_job.py
```

This runs a real local Spark job with two worker threads. The embedding service remains simulated: its four toy numbers do not encode meaning, incur API charges, or demonstrate real model latency.

`spark_job.py` creates ten tickets and distributes them across two partitions. `mapPartitionsWithIndex()` processes each partition and supplies its ID so the output can show where cache hits occur. It is the indexed variant of `mapPartitions()`.

```text
Driver: create tickets and dispatch work
    |
    +-- Partition 0: new LRU(2) -> tickets 101–105 -> enriched rows
    |
    +-- Partition 1: new LRU(2) -> tickets 106–110 -> enriched rows
    |
Driver: collect this tiny demo's results and compare embeddings
```

The job ships `lru_cache.py`, `demo.py`, and `spark_tasks.py` to the Python workers with `addPyFile()`. The processing function creates its cache inside each partition invocation, not on the driver or in shared global state.

Verified default results:

| Mode | Tickets | Simulated service calls | Hits |
|---|---|---|---|
| Uncached | 10 | 10 | 0 |
| LRU capacity 2 | 10 | 8 | 2 |

Both modes produce identical ticket IDs, text, and embeddings. The first ticket of each partition is a miss even when another partition has already processed that text. Read `spark_tasks.py` to follow each lookup, call, and insertion.

Try a larger cache or different partitioning by adding application arguments after `spark_job.py`:

```bash
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_PYTHON=/usr/bin/python3 spark-submit --master 'local[2]' --conf spark.ui.enabled=false spark_job.py --capacity 3
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_PYTHON=/usr/bin/python3 spark-submit --master 'local[2]' --conf spark.ui.enabled=false spark_job.py --partitions 5
```

The default mode compares both paths. Use `--mode cached` or `--mode uncached` to run just one. `--partitions` controls data partitions; `local[2]` controls local concurrency. They need not match.

Run all nine unit tests with `python3 -m unittest -v`. These check cache integrity, eviction, partition isolation, empty partitions, and equivalence with uncached processing. The Spark integration is separately verified by running the comparison command above.

The displayed call counts are derived from accepted output rows, not from an external service's billing logs. Retries or speculative attempts can make additional calls that do not appear in the final results. The demo collects tiny outputs on the driver; production jobs should write distributed results instead of collecting all rows. No runtime improvement is claimed for this cheap simulator.

For another machine, install a compatible Spark/Python/Java combination and set `PYSPARK_PYTHON` to a Python interpreter available to the workers. These loopback settings are for a local run; cluster workers need a reachable driver and embedding service.

References: [partition processing](https://spark.apache.org/docs/4.0.0/api/python/reference/api/pyspark.RDD.mapPartitionsWithIndex.html), [Spark execution and Python setup](https://spark.apache.org/docs/4.0.0/rdd-programming-guide.html).

## Next milestones

1. Walk through the pointer changes in `_detach()` and `_add_to_front()`.
2. Experiment with cache capacity and partition count. Partition placement and row order affect the hit rate; retries may repeat service calls.
3. Add representative input data and distributed output rather than the controlled in-memory example.
4. Compare local LRU caching with deduplicating texts, computing once per distinct input, and joining back.
5. Replace the simulator with an embedding service reachable from the Spark workers.

The local cache and Spark enrichment milestones are implemented. Keep model/settings versions in cache keys when extending it. This project is separate from `rag-learning/`.
