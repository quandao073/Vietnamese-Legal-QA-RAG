# Source this file to run the backend CLIs/API from the host venv (outside Docker):
#   source scripts/local-env.sh && cd backend && uv run python -m legalrag.ingest stats
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV=
export DATA_DIR="$ROOT/data"
export HF_HOME="$ROOT/data/models"
export QDRANT_URL=http://localhost:6333
export DATABASE_URL=postgresql+psycopg://legal:legal@localhost:5432/legalrag
export REDIS_URL=redis://localhost:6379/0
export CORS_ORIGINS=http://localhost:3000
export PYTHONUTF8=1
