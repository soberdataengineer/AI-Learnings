"""Real embeddings from an Ollama server reachable by each Spark worker."""
import json
import math
from urllib.error import URLError
from urllib.request import Request, urlopen


class OllamaEmbeddingService:
    def __init__(self, base_url="http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    def embed(self, model, text):
        request = Request(
            self.base_url + "/api/embed",
            data=json.dumps({"model": model, "input": text,
                             "truncate": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
        except URLError as error:
            raise RuntimeError(
                f"Ollama embedding request failed at {self.base_url}: {error}. "
                f"Start ollama serve and install the model with ollama pull {model}."
            ) from error
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise ValueError("Ollama must return exactly one embedding per ticket")
        vector = vectors[0]
        if (not isinstance(vector, list) or not vector or
                any(isinstance(x, bool) or not isinstance(x, (int, float))
                    or not math.isfinite(x) for x in vector)):
            raise ValueError("Ollama returned an invalid embedding")
        return vector
