"""Database engine, session factory, and FastAPI dependency for PostgreSQL.

Reads connection parameters exclusively from environment variables (.env),
never hard-coded and never placed in configs/*.yaml (SDS Section 8:
secrets live only in .env; configs/base.yaml or configs/api.yaml must not
contain any DB credential - only .env.example documents the required
keys, as already set up in Phase 2).
"""

import logging
import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("database_session")

load_dotenv()

_REQUIRED_ENV_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]


def _build_database_url() -> str:
    """Builds the PostgreSQL connection URL from environment variables.

    Returns:
        SQLAlchemy-compatible connection URL (psycopg2 driver).

    Raises:
        RuntimeError: If a required DB_* environment variable is missing
            (e.g. .env was not created from .env.example, see PHASE9_LOG.md
            hướng dẫn cài đặt PostgreSQL).
    """
    values = {name: os.getenv(name) for name in _REQUIRED_ENV_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s) in .env: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the DB_* values first "
            "(see PHASE9_LOG.md, hướng dẫn cài PostgreSQL)."
        )
    return (
        f"postgresql+psycopg2://{values['DB_USER']}:{values['DB_PASSWORD']}"
        f"@{values['DB_HOST']}:{values['DB_PORT']}/{values['DB_NAME']}"
    )


DATABASE_URL: str = _build_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in src/database/models.py."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session, closed after the request.

    Usage: ``db: Session = Depends(get_db)`` in api/routers/*.

    Yields:
        An active SQLAlchemy Session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
