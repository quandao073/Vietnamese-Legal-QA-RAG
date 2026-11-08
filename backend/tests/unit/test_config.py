from legalrag.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    s = Settings(_env_file=None)
    assert s.retrieval_top_k == 6
    assert s.chunk_strategy == "article"
    assert s.llm_model_main == "claude-sonnet-5-5"
    assert s.llm_model_small == "claude-haiku-5-5"
    assert s.embedding_model_slug == "baai-bge-m3"
    assert s.collection_name("article") == "legal_chunks__article__baai-bge-m3"


def test_settings_never_prints_key():
    s = Settings(anthropic_api_key="sk-secret", _env_file=None)
    assert "sk-secret" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-secret"


def test_cors_origins_list():
    s = Settings(anthropic_api_key="x", cors_origins="http://a:1, http://b:2", _env_file=None)
    assert s.cors_origin_list == ["http://a:1", "http://b:2"]
