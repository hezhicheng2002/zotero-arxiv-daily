"""Tests for LocalReranker; one fast fallback test and one slow real-model test."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from zotero_arxiv_daily.reranker.local import LocalReranker


def test_local_reranker_falls_back_when_primary_remote_code_breaks(config, monkeypatch):
    config.executor.debug = True
    config.reranker.local.model = "primary-model"
    config.reranker.local.fallback_model = "fallback-model"
    init_calls: list[tuple[str, dict]] = []

    class FakeSimilarity:
        def __init__(self, values):
            self._values = values

        def numpy(self):
            return self._values

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            init_calls.append((model_name, dict(kwargs)))
            if model_name == "primary-model":
                raise TypeError(
                    "AutoTokenizer.from_pretrained() got multiple values for keyword argument 'trust_remote_code'"
                )

        def encode(self, values, **kwargs):
            return np.ones((len(values), 2))

        def similarity(self, left, right):
            return FakeSimilarity(np.ones((left.shape[0], right.shape[0])))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])

    assert score.shape == (2, 1)
    assert init_calls == [
        ("primary-model", {"trust_remote_code": True}),
        ("fallback-model", {}),
    ]


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
