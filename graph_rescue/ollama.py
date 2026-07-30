from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 120

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc

    def models(self) -> list[str]:
        return [item["name"] for item in self._request("/api/tags").get("models", [])]

    def embed(self, model: str, texts: Sequence[str]) -> np.ndarray:
        response = self._request("/api/embed", {"model": model, "input": list(texts)})
        embeddings = response.get("embeddings")
        if embeddings is None:
            raise OllamaError("Ollama response did not contain embeddings")
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        format_json: bool = False,
        temperature: float = 0.0,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"
        return str(self._request("/api/generate", payload).get("response", "")).strip()


class HashingEmbedder:
    """Deterministic test fallback. It must not be used for reported experiments."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from .text import tokenize

        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index = hash(token) % self.dimensions
                vectors[row, index] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
