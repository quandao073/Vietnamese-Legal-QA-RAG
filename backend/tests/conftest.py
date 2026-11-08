from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("LEGALRAG_SKIP_WARMUP", "1")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://legal:legal@localhost:5432/legalrag")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


def _reachable(url: str, default_port: int) -> bool:
    p = urlparse(url if "//" in url else f"//{url}")
    host, port = p.hostname or "localhost", p.port or default_port
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    pg = _reachable(os.environ["DATABASE_URL"].split("@")[-1], 5432)
    qd = _reachable(os.environ["QDRANT_URL"], 6333)
    skip = pytest.mark.skip(reason="integration services not reachable (run `make infra`)")
    for item in items:
        if "integration" in item.keywords and not (pg and qd):
            item.add_marker(skip)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
