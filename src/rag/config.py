"""Settings, loaded once from .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_URL = os.environ["DATABASE_URL"]
DB_SCHEMA = os.environ.get("DB_SCHEMA", "bronze")

# If QDRANT_URL is set we talk to a Qdrant server (Docker, or a hosted cluster).
# If it is empty we run qdrant-client in embedded mode against a local folder,
# which needs no Docker at all. Switching between the two is purely this setting.
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None
QDRANT_LOCAL_PATH = PROJECT_ROOT / "qdrant_data"

COLLECTION = os.environ.get("QDRANT_COLLECTION", "meva_events")

# bge-base-en-v1.5 outputs 768-dim vectors. If you change the model you must
# change the dimension to match and re-index from scratch — a collection's
# vector size is fixed at creation.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
