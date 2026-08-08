from __future__ import annotations

from typing import Any

from aethelis.schemas.agents import AgentProfile, RelationshipRecord
from aethelis.schemas.common import AethelisModel, RecordStatus
from aethelis.schemas.events import (
    ActionProposal,
    CommittedEvent,
    EventCandidate,
    PatchOperation,
    PatchTargetType,
    StateDiff,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.ledger import BeliefCandidate, BeliefRecord, MemoryRecord
from aethelis.schemas.world import (
    AgentClaimRecord,
    Entity,
    PlayerCommitment,
    PlayerDialogueTurn,
    PlayerInventoryItem,
    PlayerKnowledgeRecord,
    PlayerRelationshipState,
    PlayerWorldResponse,
    ResourceDiscoveryState,
    WorldActivityRecord,
    WorldClockState,
    WorldResource,
    WorldState,
)


class PatchApplyResult(AethelisModel):
    patch_index: int
    applied: bool
    target_type: str
    target_id: str
    path: str
    before: Any = None
    after: Any = None
    error: str | None = None


class StateApplyReport(AethelisModel):
    applied: bool
    committed_event_id: str | None = None
    state_diff_id: str | None = None
    journal_entry_id: str | None = None
    applied_patch_count: int = 0
    skipped_patch_count: int = 0
    errors: tuple[str, ...] = ()
    patch_results: tuple[PatchApplyResult, ...] = ()

    def safe_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "committed_event_id": self.committed_event_id,
            "state_diff_id": self.state_diff_id,
            "journal_entry_id": self.journal_entry_id,
            "applied_patch_count": self.applied_patch_count,
            "skipped_patch_count": self.skipped_patch_count,
            "errors": list(self.errors),
            "patch_results": [
                {
                    "patch_index": item.patch_index,
                    "applied": item.applied,
                    "target_type": item.target_type,
                    "target_id": item.target_id,
                    "path": item.path,
                    "before": item.before,
                    "after": item.after,
                    "error": item.error,
                }
                for item in self.patch_results
            ],
        }


class ControlledStateDiffApplier:
    """Apply committed StateDiff patches to a WorldState copy.

    Public API intentionally accepts only CommittedEvent plus its
    VerificationResult. Bare ActionProposal, EventCandidate, and StateDiff are
    rejected to preserve the verification boundary.
    """

    def apply(
        self,
        *,
        world_state: WorldState,
        committed_event: CommittedEvent,
        verification_result: VerificationResult,
    ) -> tuple[WorldState, StateApplyReport]:
        if not isinstance(committed_event, CommittedEvent):
            return _not_applied(world_state, "apply requires a CommittedEvent")
        if not isinstance(verification_result, VerificationResult):
            return _not_applied(world_state, "apply requires a VerificationResult")
        if verification_result.decision != VerificationDecision.COMMIT:
            return _not_applied(
                world_state,
                f"verification decision is {verification_result.decision.value}, not commit",
                committed_event=committed_event,
            )
        if committed_event.verification_result_id != verification_result.id:
            return _not_applied(
                world_state,
                "CommittedEvent verification_result_id does not match VerificationResult id",
                committed_event=committed_event,
            )

        staged_world = world_state
        patch_results: list[PatchApplyResult] = []
        errors: list[str] = []
        for index, patch in enumerate(committed_event.state_diff.patches):
            staged_world, result = _apply_patch(staged_world, index, patch)
            patch_results.append(result)
            if result.error is not None:
                errors.append(result.error)

        if errors:
            return (
                world_state,
                StateApplyReport(
                    applied=False,
                    committed_event_id=committed_event.id,
                    state_diff_id=committed_event.state_diff.id,
                    applied_patch_count=0,
                    skipped_patch_count=len(committed_event.state_diff.patches),
                    errors=tuple(errors),
                    patch_results=tuple(_abort_applied_result(result) for result in patch_results),
                ),
            )

        applied_count = sum(1 for result in patch_results if result.applied)
        skipped_count = len(patch_results) - applied_count
        return (
            staged_world,
            StateApplyReport(
                applied=applied_count > 0 and not errors,
                committed_event_id=committed_event.id,
                state_diff_id=committed_event.state_diff.id,
                applied_patch_count=applied_count,
                skipped_patch_count=skipped_count,
                errors=tuple(errors),
                patch_results=tuple(patch_results),
            ),
        )

    def apply_state_diff(self, state_diff: StateDiff) -> None:
        raise TypeError("StateDiff cannot be applied without a committed event")

    def apply_action_proposal(self, action_proposal: ActionProposal) -> None:
        raise TypeError("ActionProposal cannot trigger StateDiff application")

    def apply_event_candidate(self, event_candidate: EventCandidate) -> None:
        raise TypeError("EventCandidate cannot trigger StateDiff application")


def _apply_patch(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    if patch.target_type == PatchTargetType.WORLD:
        if patch.operation == PatchOperation.UPDATE:
            if patch.path == "/player/current_location_id":
                return _apply_player_location_update(world_state, patch_index, patch)
            if patch.path == "/clock":
                return _apply_clock_update(world_state, patch_index, patch)
            if patch.path.startswith("/agent_") or patch.path == "/world_activities":
                return _apply_world_collection_update(world_state, patch_index, patch)
            return _apply_player_collection_update(world_state, patch_index, patch)
        if patch.operation == PatchOperation.APPEND:
            if patch.path.startswith("/agent_") or patch.path == "/world_activities":
                return _apply_world_collection_append(world_state, patch_index, patch)
            return _apply_player_collection_append(world_state, patch_index, patch)
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_patch_operation: {patch.operation.value}",
        )
    if patch.target_type == PatchTargetType.ENTITY:
        if patch.operation == PatchOperation.MARK_STATUS:
            return _apply_entity_status_mark(world_state, patch_index, patch)
        if patch.operation == PatchOperation.UPDATE:
            if patch.path.endswith("/location_id"):
                return _apply_entity_location_update(world_state, patch_index, patch)
            return _apply_entity_tags_update(world_state, patch_index, patch)
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_patch_operation: {patch.operation.value}",
        )
    if patch.target_type == PatchTargetType.RELATIONSHIP:
        if patch.operation == PatchOperation.APPEND:
            return _apply_player_collection_append(world_state, patch_index, patch)
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_patch_operation: {patch.operation.value}",
        )
    if patch.target_type == PatchTargetType.AGENT_STATE:
        if patch.operation == PatchOperation.APPEND:
            return _apply_world_collection_append(world_state, patch_index, patch)
        if patch.operation == PatchOperation.UPDATE:
            return _apply_world_collection_update(world_state, patch_index, patch)
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_patch_operation: {patch.operation.value}",
        )
    if patch.target_type != PatchTargetType.RESOURCE:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_patch_target_type: {patch.target_type.value}",
        )
    if patch.operation == PatchOperation.APPEND:
        return _apply_resource_discovery_append(world_state, patch_index, patch)
    if patch.operation in {PatchOperation.INCREMENT, PatchOperation.DECREMENT}:
        return _apply_resource_quantity_delta(world_state, patch_index, patch)
    return world_state, _patch_error(
        patch_index,
        patch,
        f"unsupported_patch_operation: {patch.operation.value}",
    )


def _apply_player_location_update(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    if patch.target_id != world_state.world_id or patch.path != "/player/current_location_id":
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    if world_state.player is None:
        return world_state, _patch_error(patch_index, patch, "player_context_missing")
    current = world_state.player.current_location_id
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    if not isinstance(patch.after, str) or patch.after not in {
        location.id for location in world_state.locations
    }:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: destination must be a known location id",
            before=current,
            after=patch.after,
        )
    updated_player = world_state.player.model_copy(update={"current_location_id": patch.after})
    return (
        world_state.model_copy(update={"player": updated_player}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=patch.after,
        ),
    )


def _apply_clock_update(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    if patch.target_id != world_state.world_id:
        return world_state, _patch_error(patch_index, patch, "clock_target_mismatch")
    current = world_state.clock.model_dump(mode="json")
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    try:
        updated = WorldClockState.model_validate(patch.after)
    except (TypeError, ValueError) as exc:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: {exc}",
            before=current,
            after=patch.after,
        )
    if updated.turn != world_state.clock.turn + 1:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: clock turn must advance exactly once",
            before=current,
            after=patch.after,
        )
    if updated.elapsed_minutes <= world_state.clock.elapsed_minutes:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: elapsed minutes must increase",
            before=current,
            after=patch.after,
        )
    return (
        world_state.model_copy(update={"clock": updated}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=updated.model_dump(mode="json"),
        ),
    )


def _world_collection_definition(path: str):
    return {
        "/agent_profiles": ("agent_profiles", AgentProfile, "id"),
        "/agent_beliefs": ("agent_beliefs", BeliefRecord, "id"),
        "/agent_memories": ("agent_memories", MemoryRecord, "id"),
        "/agent_relationships": ("agent_relationships", RelationshipRecord, "id"),
        "/agent_belief_candidates": (
            "agent_belief_candidates",
            BeliefCandidate,
            "id",
        ),
        "/agent_claims": ("agent_claims", AgentClaimRecord, "id"),
        "/world_activities": ("world_activities", WorldActivityRecord, "id"),
    }.get(path)


def _apply_world_collection_append(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    return _apply_world_collection(
        world_state,
        patch_index,
        patch,
        require_nonshrinking=True,
        require_prefix=True,
    )


def _apply_world_collection_update(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    return _apply_world_collection(
        world_state,
        patch_index,
        patch,
        require_nonshrinking=False,
        require_prefix=False,
    )


def _apply_world_collection(
    world_state: WorldState,
    patch_index: int,
    patch,
    *,
    require_nonshrinking: bool,
    require_prefix: bool,
) -> tuple[WorldState, PatchApplyResult]:
    definition = _world_collection_definition(patch.path)
    if definition is None:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    field_name, model_type, identity_field = definition
    current_records = getattr(world_state, field_name)
    current = [record.model_dump(mode="json") for record in current_records]
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    if not isinstance(patch.after, list):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: world collection must be a list",
            before=current,
            after=patch.after,
        )
    try:
        updated_records = tuple(model_type.model_validate(item) for item in patch.after)
    except (TypeError, ValueError) as exc:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: {exc}",
            before=current,
            after=patch.after,
        )
    identities = [getattr(record, identity_field) for record in updated_records]
    if len(set(identities)) != len(identities):
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: duplicate {field_name} identity",
            before=current,
            after=patch.after,
        )
    if require_nonshrinking and len(updated_records) < len(current_records):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: append cannot remove world records",
            before=current,
            after=patch.after,
        )
    if require_prefix and tuple(updated_records[: len(current_records)]) != current_records:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: append cannot rewrite existing world records",
            before=current,
            after=patch.after,
        )
    after = [record.model_dump(mode="json") for record in updated_records]
    return (
        world_state.model_copy(update={field_name: updated_records}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=after,
        ),
    )


def _apply_player_collection_append(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    if world_state.player is None:
        return world_state, _patch_error(patch_index, patch, "player_context_missing")
    collections = {
        "/player/knowledge": ("knowledge", PlayerKnowledgeRecord, "id"),
        "/player/relationships": (
            "relationships",
            PlayerRelationshipState,
            "character_id",
        ),
        "/player/dialogue_history": ("dialogue_history", PlayerDialogueTurn, "id"),
        "/player/inventory": ("inventory", PlayerInventoryItem, "id"),
        "/player/commitments": ("commitments", PlayerCommitment, "id"),
        "/player/world_responses": ("world_responses", PlayerWorldResponse, "id"),
    }
    definition = collections.get(patch.path)
    if definition is None:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    field_name, model_type, identity_field = definition
    current_records = getattr(world_state.player, field_name)
    current = [record.model_dump(mode="json") for record in current_records]
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    if not isinstance(patch.after, list):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: player collection must be a list",
            before=current,
            after=patch.after,
        )
    try:
        updated_records = tuple(model_type.model_validate(item) for item in patch.after)
    except (TypeError, ValueError) as exc:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: {exc}",
            before=current,
            after=patch.after,
        )
    identities = [getattr(record, identity_field) for record in updated_records]
    if len(set(identities)) != len(identities):
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: duplicate {field_name} identity",
            before=current,
            after=patch.after,
        )
    if len(updated_records) < len(current_records):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: append cannot remove player records",
            before=current,
            after=patch.after,
        )
    updated_player = world_state.player.model_copy(update={field_name: updated_records})
    after = [record.model_dump(mode="json") for record in updated_records]
    return (
        world_state.model_copy(update={"player": updated_player}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=after,
        ),
    )


def _apply_player_collection_update(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    if world_state.player is None:
        return world_state, _patch_error(patch_index, patch, "player_context_missing")
    collections = {
        "/player/inventory": ("inventory", PlayerInventoryItem, "id"),
        "/player/commitments": ("commitments", PlayerCommitment, "id"),
    }
    definition = collections.get(patch.path)
    if definition is None:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    field_name, model_type, identity_field = definition
    current_records = getattr(world_state.player, field_name)
    current = [record.model_dump(mode="json") for record in current_records]
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    if not isinstance(patch.after, list):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: player collection must be a list",
            before=current,
            after=patch.after,
        )
    try:
        updated_records = tuple(model_type.model_validate(item) for item in patch.after)
    except (TypeError, ValueError) as exc:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: {exc}",
            before=current,
            after=patch.after,
        )
    identities = [getattr(record, identity_field) for record in updated_records]
    if len(set(identities)) != len(identities):
        return world_state, _patch_error(
            patch_index,
            patch,
            f"invalid_patch_after: duplicate {field_name} identity",
            before=current,
            after=patch.after,
        )
    updated_player = world_state.player.model_copy(update={field_name: updated_records})
    after = [record.model_dump(mode="json") for record in updated_records]
    return (
        world_state.model_copy(update={"player": updated_player}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=after,
        ),
    )


def _apply_entity_tags_update(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    expected_path = f"/entity/{patch.target_id}/tags"
    if patch.path != expected_path:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    entities = list(world_state.entities)
    entity_index = _entity_index(entities, patch.target_id)
    if entity_index is None:
        return world_state, _patch_error(patch_index, patch, "entity_target_not_found")
    entity = entities[entity_index]
    current = list(entity.tags)
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    if not isinstance(patch.after, list) or not all(
        isinstance(item, str) and item for item in patch.after
    ):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: entity tags must be a list of non-empty strings",
            before=current,
            after=patch.after,
        )
    updated_entity = entity.model_copy(update={"tags": tuple(dict.fromkeys(patch.after))})
    entities[entity_index] = updated_entity
    return (
        world_state.model_copy(update={"entities": tuple(entities)}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=list(updated_entity.tags),
        ),
    )


def _apply_entity_location_update(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    expected_path = f"/entity/{patch.target_id}/location_id"
    if patch.path != expected_path:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    entities = list(world_state.entities)
    entity_index = _entity_index(entities, patch.target_id)
    if entity_index is None:
        return world_state, _patch_error(patch_index, patch, "entity_target_not_found")
    entity = entities[entity_index]
    current = entity.location_id
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    location_ids = {location.id for location in world_state.locations}
    if patch.after is not None and patch.after not in location_ids:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: entity destination must be a known location",
            before=current,
            after=patch.after,
        )
    updated_entity = entity.model_copy(update={"location_id": patch.after})
    entities[entity_index] = updated_entity
    return (
        world_state.model_copy(update={"entities": tuple(entities)}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=patch.after,
        ),
    )


def _apply_entity_status_mark(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    expected_path = f"/entity/{patch.target_id}/status"
    if patch.path != expected_path:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )

    entities = list(world_state.entities)
    entity_index = _entity_index(entities, patch.target_id)
    if entity_index is None:
        return world_state, _patch_error(patch_index, patch, "entity_target_not_found")

    entity = entities[entity_index]
    current = entity.status.value
    if _status_value(patch.before) != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )

    after_status = _record_status(patch.after)
    if after_status is None:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: entity status must be a valid RecordStatus",
            before=current,
            after=patch.after,
        )

    updated_entity = entity.model_copy(update={"status": after_status})
    entities[entity_index] = updated_entity
    return (
        world_state.model_copy(update={"entities": tuple(entities)}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=updated_entity.status.value,
        ),
    )


def _apply_resource_discovery_append(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    expected_path = f"/resource/{patch.target_id}/discovery_state/discovered_by_agent_ids"
    if patch.path != expected_path:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )

    resources = list(world_state.resources)
    resource_index = _resource_index(resources, patch.target_id)
    if resource_index is None:
        return world_state, _patch_error(patch_index, patch, "resource_target_not_found")

    resource = resources[resource_index]
    current = list(resource.discovery_state.discovered_by_agent_ids)
    if patch.before != current:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {current}",
            before=current,
            after=patch.after,
        )
    if not isinstance(patch.after, list) or not all(isinstance(item, str) for item in patch.after):
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: append discovery after must be a list of agent ids",
            before=current,
            after=patch.after,
        )

    updated_resource = resource.model_copy(
        update={
            "discovery_state": ResourceDiscoveryState(
                discovered_by_agent_ids=tuple(dict.fromkeys(patch.after))
            )
        }
    )
    resources[resource_index] = updated_resource
    updated_world = world_state.model_copy(update={"resources": tuple(resources)})
    return (
        updated_world,
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=current,
            after=list(updated_resource.discovery_state.discovered_by_agent_ids),
        ),
    )


def _apply_resource_quantity_delta(
    world_state: WorldState,
    patch_index: int,
    patch,
) -> tuple[WorldState, PatchApplyResult]:
    expected_path = f"/resource/{patch.target_id}/quantity"
    if patch.path != expected_path:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"unsupported_target_path: {patch.path}",
        )
    resources = list(world_state.resources)
    resource_index = _resource_index(resources, patch.target_id)
    if resource_index is None:
        return world_state, _patch_error(patch_index, patch, "resource_target_not_found")

    resource = resources[resource_index]
    if patch.before != resource.quantity:
        return world_state, _patch_error(
            patch_index,
            patch,
            f"patch_before_mismatch: expected {resource.quantity}",
            before=resource.quantity,
            after=patch.after,
        )
    if not isinstance(patch.after, int) or patch.after < 0:
        return world_state, _patch_error(
            patch_index,
            patch,
            "invalid_patch_after: resource quantity must be a non-negative integer",
            before=resource.quantity,
            after=patch.after,
        )

    updated_resource = resource.model_copy(update={"quantity": patch.after})
    resources[resource_index] = updated_resource
    return (
        world_state.model_copy(update={"resources": tuple(resources)}),
        PatchApplyResult(
            patch_index=patch_index,
            applied=True,
            target_type=patch.target_type.value,
            target_id=patch.target_id,
            path=patch.path,
            before=resource.quantity,
            after=updated_resource.quantity,
        ),
    )


def _abort_applied_result(result: PatchApplyResult) -> PatchApplyResult:
    if not result.applied:
        return result
    return result.model_copy(update={"applied": False, "error": "aborted_due_to_patch_error"})


def _not_applied(
    world_state: WorldState,
    error: str,
    *,
    committed_event: CommittedEvent | None = None,
) -> tuple[WorldState, StateApplyReport]:
    return (
        world_state,
        StateApplyReport(
            applied=False,
            committed_event_id=committed_event.id if committed_event is not None else None,
            state_diff_id=committed_event.state_diff.id if committed_event is not None else None,
            errors=(error,),
        ),
    )


def _patch_error(
    patch_index: int,
    patch,
    error: str,
    *,
    before: Any = None,
    after: Any = None,
) -> PatchApplyResult:
    return PatchApplyResult(
        patch_index=patch_index,
        applied=False,
        target_type=patch.target_type.value,
        target_id=patch.target_id,
        path=patch.path,
        before=before,
        after=after,
        error=error,
    )


def _resource_index(resources: list[WorldResource], resource_id: str) -> int | None:
    for index, resource in enumerate(resources):
        if resource.id == resource_id:
            return index
    return None


def _entity_index(entities: list[Entity], entity_id: str) -> int | None:
    for index, entity in enumerate(entities):
        if entity.id == entity_id:
            return index
    return None


def _record_status(value) -> RecordStatus | None:
    if isinstance(value, RecordStatus):
        return value
    if isinstance(value, str):
        try:
            return RecordStatus(value)
        except ValueError:
            return None
    return None


def _status_value(value) -> str | None:
    status = _record_status(value)
    return status.value if status is not None else None
