"""Standalone script to verify the PostgreSQL connection configured via
.env, before running Alembic migrations or the API for the first time.

Usage (từ project root, venv đã activate):
    python scripts\\check_db_connection.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from src.database.session import engine


def main() -> None:
    """Tries a trivial query and reports success/failure in plain language."""
    try:
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version();")).scalar()
        print("Ket noi PostgreSQL thanh cong.")
        print(f"Server version: {version}")
    except Exception as error:  # noqa: BLE001 - diagnostic script, surface any failure as-is
        print("Ket noi PostgreSQL that bai.")
        print(f"Loi: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
