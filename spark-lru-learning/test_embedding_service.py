import io
import json
import unittest
from unittest.mock import patch

from embedding_service import OllamaEmbeddingService
from spark_tasks import process_partition


class EmbeddingTests(unittest.TestCase):
    @patch("embedding_service.urlopen")
    def test_request_and_cache_reuse(self, request):
        request.side_effect = lambda *a, **k: io.BytesIO(
            json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode())
        rows = list(process_partition(
            0, iter([(1, "Payment failed"), (2, "Payment failed")]),
            provider="ollama"))
        self.assertEqual(request.call_count, 1)
        self.assertEqual([r["cache_hit"] for r in rows], [False, True])
        self.assertEqual(rows[0]["embedding"], rows[1]["embedding"])
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload, {"model": "nomic-embed-text",
                                   "input": "Payment failed", "truncate": False})

    @patch("embedding_service.urlopen")
    def test_invalid_vectors_are_rejected(self, request):
        for vectors in [[], [[]], [["bad"]], [[float("nan")]], [[1], [2]]]:
            with self.subTest(vectors=vectors):
                request.side_effect = lambda *a, **k: io.BytesIO(
                    json.dumps({"embeddings": vectors}).encode())
                with self.assertRaises(ValueError):
                    OllamaEmbeddingService().embed("model", "text")
