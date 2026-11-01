import unittest

from spark_tasks import process_partition


class PartitionTests(unittest.TestCase):
    def test_cache_preserves_every_ticket_and_reduces_calls(self):
        tickets = [(1, "Return"), (2, "Payment"), (3, "Return"),
                   (4, "Order"), (5, "Payment")]
        cached = list(process_partition(0, iter(tickets), capacity=2))
        baseline = list(process_partition(0, iter(tickets), use_cache=False))
        for actual, expected in zip(cached, baseline):
            for field in ["ticket_id", "text", "embedding", "partition_id"]:
                self.assertEqual(actual[field], expected[field])
        self.assertEqual(len(cached), len(tickets))
        self.assertEqual(sum(row["service_calls"] for row in cached), 4)
        self.assertEqual(sum(row["service_calls"] for row in baseline), 5)
        self.assertEqual([row["cache_hit"] for row in cached], [False, False, True, False, False])

    def test_partitions_do_not_share_cache(self):
        tickets = [(1, "Return"), (2, "Return")]
        for partition_id in [0, 1]:
            rows = list(process_partition(partition_id, iter(tickets)))
            self.assertEqual([row["cache_hit"] for row in rows], [False, True])
            self.assertTrue(all(row["partition_id"] == partition_id for row in rows))

    def test_empty_partition(self):
        self.assertEqual(list(process_partition(0, iter([]))), [])

    def test_capacity_changes_eviction(self):
        tickets = list(enumerate(["A", "B", "A"]))
        small = list(process_partition(0, iter(tickets), capacity=1))
        large = list(process_partition(0, iter(tickets), capacity=2))
        self.assertEqual(sum(row["service_calls"] for row in small), 3)
        self.assertEqual(sum(row["service_calls"] for row in large), 2)


if __name__ == "__main__":
    unittest.main()
