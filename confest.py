"""
Root conftest.py
Adds the project root to sys.path so that top-level packages
(db, gateway, config, etc.) are importable when running pytest
from the project root.
"""

import os
import sys

# Insert the project root (the directory this file lives in) at the
# front of sys.path so 'db', 'gateway', 'config', etc. are importable.
sys.path.insert(0, os.path.dirname(__file__))


# Normalize DATABASE_URL — treat missing or empty string as "not set".
# Local dev  : falls back to SQLite so no Postgres needed.
# CI         : DATABASE_URL is set in the workflow env, Postgres is used.
db_url = os.environ.get("DATABASE_URL", "").strip()
if not db_url:
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
