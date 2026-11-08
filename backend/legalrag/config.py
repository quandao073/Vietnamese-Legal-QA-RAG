"""Application settings loaded from environment / .env (pydantic-settings)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- Anthropic ---
    anthropic_api_key: SecretStr = SecretStr("")
    llm_model_main: str = "claude-sonnet-5-5"
    llm_model_small: str = "claude-haiku-5-5"
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.1  # kept for compatibility; not sent (ADR-001)
    llm_max_concurrency: int = 4
    llm_cache_enabled: bool = False
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0

    # --- Models ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True
    device: Literal["cpu", "cuda"] = "cpu"
    hf_home: str = "/data/models"

    # --- Retrieval ---
    chunk_strategy: str = "article"
    retrieval_top_n_dense: int = 30
    retrieval_top_n_bm25: int = 30
    retrieval_rrf_k: int = 60
    retrieval_top_k: int = 6
    context_token_budget: int = 6000
    query_rewrite_enabled: bool = True
    min_rerank_score: float = 0.2
    citation_llm_verify: bool = False

    # --- Ingest ---
    corpus_max_docs: int = 20000
    eval_yuitc_sample: int = 500
    max_chunk_tokens: int = 700
    random_seed: int = 42
    data_dir: str = "/data"

    # --- Infra ---
    qdrant_url: str = "http://qdrant:6333"
    database_url: str = "postgresql+psycopg://legal:legal@postgres:5432/legalrag"
    redis_url: str = "redis://redis:6379/0"
    admin_token: SecretStr = SecretStr("change-me")
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"
    rate_limit_per_minute: int = 30

    @property
    def embedding_model_slug(self) -> str:
        return slugify(self.embedding_model)

    def collection_name(self, strategy: str, embedding_model: str | None = None) -> str:
        slug = slugify(embedding_model) if embedding_model else self.embedding_model_slug
        return f"legal_chunks__{strategy}__{slug}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings", "slugify", "Field"]
