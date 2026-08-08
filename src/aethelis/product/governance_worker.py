from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

from aethelis.product.command_contracts import CommandResultView, GovernanceWorkItem
from aethelis.product.governance_contracts import GovernedWorldOutcome
from aethelis.product.world_engine import ProductWorldEngine


class GovernanceRepository(Protocol):
    def claim_next_governance(
        self, *, worker_id: str, now: datetime, lease_duration: timedelta
    ) -> GovernanceWorkItem | None: ...

    def commit_governed_outcome(
        self,
        *,
        item: GovernanceWorkItem,
        outcome: GovernedWorldOutcome,
        worker_id: str,
        now: datetime,
    ) -> CommandResultView: ...


class GovernanceWorker:
    def __init__(
        self,
        repository: GovernanceRepository,
        *,
        worker_id: str,
        engine: ProductWorldEngine | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        lease_duration: timedelta = timedelta(seconds=90),
    ) -> None:
        self._repository = repository
        self._worker_id = worker_id
        self._engine = engine or ProductWorldEngine()
        self._clock = clock
        self._lease_duration = lease_duration

    def run_once(self) -> CommandResultView | None:
        item = self._repository.claim_next_governance(
            worker_id=self._worker_id,
            now=self._clock(),
            lease_duration=self._lease_duration,
        )
        if item is None:
            return None
        if item.execution.parsed_intent is None:
            raise RuntimeError("governance claim did not contain a parsed intent")
        outcome = self._engine.govern(
            command=item.command,
            intent=item.execution.parsed_intent,
            world_state=item.world_state,
            content_package=item.content_package,
        )
        return self._repository.commit_governed_outcome(
            item=item,
            outcome=outcome,
            worker_id=self._worker_id,
            now=self._clock(),
        )
