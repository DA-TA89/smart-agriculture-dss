"""Alembic environment script.

Builds the DB URL from .env (never hard-coded, SDS Section 8) via the
exact same _build_database_url() used by the running API (single source
of truth - no separate/duplicated connection-string logic for
migrations vs. runtime). Points Alembic's autogenerate at
src.database.session.Base.metadata so ``alembic revision --autogenerate``
picks up every model in src/database/models.py automatically.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# --- sys.path bootstrap (same pattern as other pipelines/*.py in this project) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.models import PredictionRecord  # noqa: F401 - registers table with Base.metadata
from src.database.session import Base, _build_database_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", _build_database_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generates SQL scripts without a live DB connection (not used in this project's workflow)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Runs migrations against a live connection - the normal workflow used here."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
