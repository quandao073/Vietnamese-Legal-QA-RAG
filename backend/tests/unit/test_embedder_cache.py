import numpy as np

from legalrag.cache import MemoryCache
from legalrag.retrieval.embedder import CachedQueryEmbedder, FakeEmbedder


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder(dim=32)
    a = e.encode_query("thuế giá trị gia tăng")
    b = e.encode_query("thuế giá trị gia tăng")
    assert a.shape == (32,) and np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    m = e.encode_passages(["a", "b c"])
    assert m.shape == (2, 32)
    assert e.encode_query("thuế") @ e.encode_query("thuế suất") > e.encode_query(
        "thuế"
    ) @ e.encode_query("hôn nhân")


def test_query_cache_hit_skips_model():
    inner = FakeEmbedder(dim=16)
    cache = MemoryCache()
    emb = CachedQueryEmbedder(inner, cache, model_name="fake", ttl_seconds=10)
    v1 = emb.encode_query("câu hỏi")
    calls_after_first = inner.query_calls
    v2 = emb.encode_query("câu hỏi")
    assert inner.query_calls == calls_after_first == 1
    assert np.allclose(v1, v2) and v2.dtype == np.float32
    assert emb.dim == 16


def test_memory_cache_ttl_and_miss():
    c = MemoryCache()
    assert c.get("k") is None
    c.set("k", b"v", ttl=60)
    assert c.get("k") == b"v"
