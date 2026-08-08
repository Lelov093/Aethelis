from __future__ import annotations

import time
from uuid import uuid4

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aethelis.config.settings import DEFAULT_ENV_FILE, load_settings
from aethelis.db.command_repository import SQLAlchemyCommandRepository
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.llm.openai_compatible import OpenAICompatibleLLMProvider
from aethelis.product.command_worker import CommandWorker, StructuredIntentParser
from aethelis.product.dialogue_expression import DialogueExpressionService
from aethelis.product.governance_worker import GovernanceWorker
from aethelis.product.world_engine import ProductWorldEngine


class WorkerRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHELIS_WORKER_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    id: str = Field(default_factory=lambda: f"worker_{uuid4().hex}", min_length=1)
    poll_seconds: float = Field(default=1.0, gt=0, le=60)


def main() -> None:
    runtime = WorkerRuntimeSettings()
    engine = create_db_engine(load_database_settings())
    provider = OpenAICompatibleLLMProvider(load_settings())
    worker = CommandWorker(
        SQLAlchemyCommandRepository(engine),
        StructuredIntentParser(provider),
        worker_id=runtime.id,
    )
    governance = GovernanceWorker(
        SQLAlchemyCommandRepository(engine),
        worker_id=runtime.id,
        engine=ProductWorldEngine(DialogueExpressionService(provider)),
    )
    try:
        while True:
            parsed = worker.run_once()
            governed = governance.run_once()
            if parsed is None and governed is None:
                time.sleep(runtime.poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
        engine.dispose()
