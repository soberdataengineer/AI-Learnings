"""Demonstrate repeated ticket text with a deterministic simulated service."""
import hashlib

from lru_cache import LRUCache


class SimulatedEmbeddingService:
    def __init__(self):
        self.calls = 0

    def embed(self, model, text):
        self.calls += 1
        # Stable toy numbers for testing only; these do NOT encode meaning.
        digest = hashlib.sha256((model + "\0" + text).encode()).digest()
        return [round(byte / 255, 4) for byte in digest[:4]]


def main():
    tickets = [
        (101, "Reset my password"),
        (102, "Payment failed"),
        (103, "Reset my password"),
        (104, "Account locked"),
        (105, "Payment failed"),
    ]
    cache = LRUCache(capacity=2)
    service = SimulatedEmbeddingService()
    model = "simulated-v1"

    for ticket_id, text in tickets:
        key = (model, text)
        try:
            vector = cache.get(key)
            status = "HIT"
        except KeyError:
            vector = service.embed(model, text)
            cache.put(key, vector)
            status = "MISS"

        order = [key[1] for key in cache.keys_mru_to_lru()]
        print(f"{ticket_id} | {status:4} | {text}")
        print(f"  vector: {vector}")
        print(f"  most recent -> least recent: {order}")

    print(f"\nTickets: {len(tickets)} | Service calls: {service.calls} | Cache hits: {len(tickets) - service.calls}")


if __name__ == "__main__":
    main()
