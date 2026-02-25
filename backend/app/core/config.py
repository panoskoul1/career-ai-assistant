"""Application configuration loaded from environment variables.

After the architecture refactoring:
  - The backend does NOT load ML models or access Qdrant directly.
  - All AI compute is delegated to Nuclio functions via HTTP.
  - FN_INGEST_URL → fn-ingest (chunking + embedding + storage)
  - NUCLIO_URL    → fn-agent  (intent classification + ReActAgent)
"""

from __future__ import annotations

import os


class Settings:
    # Nuclio function endpoints
    FN_INGEST_URL: str = os.getenv("FN_INGEST_URL", "http://localhost:9090")
    NUCLIO_URL: str = os.getenv("NUCLIO_URL", "http://localhost:9091")   # fn-agent

    # Document storage
    RESUME_COLLECTION: str = "resume_chunks"

    # Request limits
    MAX_UPLOAD_SIZE_MB: int = 10
    MAX_MESSAGE_LENGTH: int = 2000


settings = Settings()
