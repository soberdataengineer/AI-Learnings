"""Run a small support-ticket embedding job with partition-local LRU caches."""
import argparse
import math
from functools import partial
from pathlib import Path

from pyspark.sql import SparkSession

from spark_tasks import process_partition


def sample_tickets():
    # Two identical sequences with distinct IDs. With two slices, each partition
    # receives five tickets in this controlled example. No shuffle is applied.
    texts = [
        "How do I return an item?",
        "My payment failed",
        "How do I return an item?",
        "Where is my order?",
        "My payment failed",
    ]
    return [(101 + i, text) for i, text in enumerate(texts * 2)]


def run_job(tickets, capacity, use_cache, provider="simulated", model=None,
            ollama_url="http://127.0.0.1:11434"):
    processor = partial(process_partition, capacity=capacity, use_cache=use_cache,
                        provider=provider, model=model, ollama_url=ollama_url)
    # Same as mapPartitions, with an extra partition index for explaining results.
    return tickets.mapPartitionsWithIndex(processor).collect()


def print_results(label, rows):
    print(f"\n{label}")
    print("ticket | partition | result | text")
    for row in sorted(rows, key=lambda row: row["ticket_id"]):
        status = "HIT" if row["cache_hit"] else "CALL"
        print(f"{row['ticket_id']} | {row['partition_id']} | {status:4} | {row['text']}")
    calls = sum(row["service_calls"] for row in rows)
    hits = sum(row["cache_hit"] for row in rows)
    print(f"Tickets: {len(rows)} | Service calls: {calls} | Cache hits: {hits}")
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--partitions", type=int, default=2)
    parser.add_argument("--mode", choices=["compare", "cached", "uncached"], default="compare")
    parser.add_argument("--provider", choices=["ollama", "simulated"], default="ollama")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    if args.capacity < 1 or args.partitions < 1:
        parser.error("capacity and partitions must be positive")

    # spark-submit controls the master, e.g. --master 'local[2]'.
    spark = SparkSession.builder.appName("SupportTicketLRU").getOrCreate()
    try:
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        root = Path(__file__).resolve().parent
        # Ship helper modules so Python workers can import them.
        for name in ["lru_cache.py", "demo.py", "embedding_service.py", "spark_tasks.py"]:
            sc.addPyFile(str(root / name))

        tickets = sc.parallelize(sample_tickets(), args.partitions)
        print(f"Spark {spark.version} | Partitions: {tickets.getNumPartitions()} | Capacity: {args.capacity}")
        print(f"Embedding provider: {args.provider}")
        embedding_options = dict(provider=args.provider, model=args.embedding_model,
                                 ollama_url=args.ollama_url)

        if args.mode in {"compare", "uncached"}:
            baseline = run_job(tickets, args.capacity, use_cache=False, **embedding_options)
            baseline_calls = print_results("UNCACHED", baseline)
        if args.mode in {"compare", "cached"}:
            cached = run_job(tickets, args.capacity, use_cache=True, **embedding_options)
            cached_calls = print_results("LRU CACHED", cached)
        if args.mode == "compare":
            # Compare ticket identity, text, and embedding; order and metrics differ.
            expected = {row["ticket_id"]: (row["text"], row["embedding"]) for row in baseline}
            actual = {row["ticket_id"]: (row["text"], row["embedding"]) for row in cached}
            equivalent = actual.keys() == expected.keys() and all(
                actual[key][0] == expected[key][0]
                and len(actual[key][1]) == len(expected[key][1])
                and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
                        for a, b in zip(actual[key][1], expected[key][1]))
                for key in expected)
            if len(cached) != len(baseline) or not equivalent:
                raise AssertionError("Cached output differs from the uncached output")
            print("\nPASS: cached and uncached ticket embeddings match within numerical tolerance.")
            print(f"Service calls avoided in accepted task results: {baseline_calls - cached_calls}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
