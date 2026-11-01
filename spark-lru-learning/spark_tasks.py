"""Partition processing, also runnable in unit tests without Spark."""
from demo import SimulatedEmbeddingService
from lru_cache import LRUCache
from embedding_service import OllamaEmbeddingService


def process_partition(partition_id, tickets, capacity=2, use_cache=True,
                      provider="simulated", model=None,
                      ollama_url="http://127.0.0.1:11434"):
    # Created here, rather than on the driver: each invocation starts empty.
    cache = LRUCache(capacity) if use_cache else None
    if provider == "ollama":
        service = OllamaEmbeddingService(ollama_url)
        model = model or "nomic-embed-text"
    elif provider == "simulated":
        service = SimulatedEmbeddingService()
        model = model or "simulated-v1"
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    for ticket_id, text in tickets:
        key = (model, text)
        cache_hit = False
        if cache is not None:
            try:
                vector = cache.get(key)
                cache_hit = True
            except KeyError:
                pass

        if not cache_hit:
            vector = service.embed(model, text)
            if cache is not None:
                cache.put(key, vector)

        yield {
            "ticket_id": ticket_id,
            "text": text,
            "embedding": vector,
            "partition_id": partition_id,
            "cache_hit": cache_hit,
            "service_calls": 0 if cache_hit else 1,
        }
