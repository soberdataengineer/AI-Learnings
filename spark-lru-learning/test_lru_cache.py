import unittest

from demo import SimulatedEmbeddingService
from lru_cache import LRUCache


class LRUCacheTests(unittest.TestCase):
    def assert_links(self, cache):
        """Verify dictionary/list agreement and bidirectional links."""
        previous = cache.head
        node = previous.next
        visited = set()
        while node is not cache.tail:
            self.assertNotIn(node.key, visited)
            visited.add(node.key)
            self.assertIs(node.prev, previous)
            self.assertIs(cache.nodes[node.key], node)
            previous, node = node, node.next
        self.assertIs(cache.tail.prev, previous)
        self.assertEqual(visited, set(cache.nodes))

    def test_read_changes_eviction_candidate(self):
        cache = LRUCache(2)
        cache.put("A", 10)
        cache.put("B", 20)
        self.assertEqual(cache.get("A"), 10)
        cache.put("C", 30)
        with self.assertRaises(KeyError):
            cache.get("B")
        self.assertEqual(cache.keys_mru_to_lru(), ["C", "A"])
        self.assert_links(cache)

    def test_update_reuses_node_and_does_not_evict(self):
        cache = LRUCache(2)
        cache.put("A", 10)
        original_node = cache.nodes["A"]
        cache.put("B", 20)
        cache.put("A", 99)
        self.assertIs(cache.nodes["A"], original_node)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.get("A"), 99)
        self.assertEqual(cache.keys_mru_to_lru(), ["A", "B"])
        self.assert_links(cache)

    def test_none_is_a_value_and_miss_does_not_change_order(self):
        cache = LRUCache(1)
        cache.put("A", None)
        self.assertIsNone(cache.get("A"))
        with self.assertRaises(KeyError):
            cache.get("missing")
        self.assertEqual(cache.keys_mru_to_lru(), ["A"])
        cache.put("B", 2)
        self.assertEqual(cache.keys_mru_to_lru(), ["B"])
        self.assert_links(cache)

    def test_capacity_validation(self):
        for capacity in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                LRUCache(capacity)

    def test_ticket_results_match_uncached_computation(self):
        texts = ["Reset", "Payment", "Reset", "Locked", "Payment"]
        service = SimulatedEmbeddingService()
        baseline = SimulatedEmbeddingService()
        cache = LRUCache(2)
        results = []
        for text in texts:
            key = ("v1", text)
            try:
                vector = cache.get(key)
            except KeyError:
                vector = service.embed(*key)
                cache.put(key, vector)
            results.append(vector)
            self.assert_links(cache)
        self.assertEqual(results, [baseline.embed("v1", text) for text in texts])
        self.assertEqual(service.calls, 4)
        self.assertEqual(baseline.calls, 5)


if __name__ == "__main__":
    unittest.main()
