from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from aethelis.config.settings import DEFAULT_ENV_FILE
from aethelis.utils.redaction import redact_text


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: SecretStr


class DatabaseConfigurationError(ValueError):
    """Safe database configuration error."""


def load_database_settings(env_file: Path | str | None = DEFAULT_ENV_FILE) -> DatabaseSettings:
    resolved = Path(env_file) if env_file is not None else None
    if resolved is not None and not resolved.is_file():
        raise DatabaseConfigurationError(f"Configuration file not found: {resolved}")
    try:
        return DatabaseSettings(_env_file=resolved)
    except (ValidationError, SettingsError) as exc:
        raise DatabaseConfigurationError(
            f"Invalid database configuration. Check DATABASE_URL in .env.\n{redact_text(str(exc))}"
        ) from None


def create_db_engine(settings: DatabaseSettings) -> Engine:
    return create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)


def check_database_health(engine: Engine) -> dict[str, object]:
    with engine.connect() as connection:
        database = connection.execute(text("select current_database()")).scalar_one()
        version = connection.execute(text("select version()")).scalar_one()
        pgvector_available = bool(
            connection.execute(
                text("select exists(select 1 from pg_available_extensions where name = 'vector')")
            ).scalar_one()
        )
        pgvector_installed = bool(
            connection.execute(
                text("select exists(select 1 from pg_extension where extname = 'vector')")
            ).scalar_one()
        )
        vector_extension_version = connection.execute(
            text("select extversion from pg_extension where extname = 'vector'")
        ).scalar_one_or_none()
        embedding_vector_column = connection.execute(
            text(
                """
                select format_type(a.atttypid, a.atttypmod)
                from pg_attribute a
                join pg_class c on c.oid = a.attrelid
                join pg_namespace n on n.oid = c.relnamespace
                where n.nspname = 'public'
                  and c.relname = 'embedding_chunks'
                  and a.attname = 'embedding_vector'
                  and not a.attisdropped
                """
            )
        ).scalar_one_or_none()
    return {
        "database": database,
        "postgresql": str(version).split(" on ", 1)[0],
        "pgvector_available": pgvector_available,
        "pgvector_installed": pgvector_installed,
        "vector_extension_version": vector_extension_version,
        "embedding_vector_column": embedding_vector_column,
        "embedding_vector_column_ready": embedding_vector_column == "vector(1024)",
    }
