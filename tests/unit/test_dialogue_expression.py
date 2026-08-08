from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from aethelis.llm.structured import parse_structured_output
from aethelis.product.command_contracts import (
    CommandInputMode,
    ParsedPlayerIntent,
    PlayerCommand,
)
from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.dialogue_expression import DialogueExpressionService
from aethelis.product.world_engine import ProductWorldEngine
from aethelis.providers import ProviderAttempt

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 3, 15, 0, tzinfo=UTC)


def command(
    *,
    location_id: str,
    action_id: str,
    target_ids: tuple[str, ...],
) -> PlayerCommand:
    return PlayerCommand(
        id="command_expression",
        idempotency_key="expression-request-0001",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="instance_1",
        play_session_id="session_1",
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id=action_id,
        actor_id="profile_1",
        target_ids=target_ids,
        location_id=location_id,
        expected_world_version=0,
        locale="zh-CN",
        submitted_at=NOW,
        updated_at=NOW,
    )


class _StructuredProvider:
    provider_name = "structured_test_provider"

    def __init__(
        self,
        outputs: tuple[dict[str, object], ...],
        *,
        latency_ms: int = 10,
        total_tokens: int = 15,
    ) -> None:
        self._outputs = iter(outputs)
        self._latency_ms = latency_ms
        self._total_tokens = total_tokens
        self.call_count = 0

    def generate_structured(self, _prompt, schema_type, **_kwargs):
        self.call_count += 1
        raw_text = json.dumps(next(self._outputs), ensure_ascii=False)
        return parse_structured_output(
            raw_text=raw_text,
            schema_type=schema_type,
            model_name=f"test-model-{self.call_count}",
            provider_name=self.provider_name,
            latency_ms=self._latency_ms,
            usage={"total_tokens": self._total_tokens},
            attempts=(
                ProviderAttempt(
                    f"test-model-{self.call_count}",
                    True,
                    self._latency_ms,
                ),
            ),
        )


def test_reviewed_provider_expression_is_persisted_without_raw_output() -> None:
    utterance = "塞尔卡压低声音：零件只剩几组，工坊和议会都在争取。"
    candidate_hash = sha256(utterance.encode("utf-8")).hexdigest()
    provider = _StructuredProvider(
        (
            {
                "character_id": "selka",
                "dialogue_option_id": "ask_selka_about_parts",
                "utterance": utterance,
                "supported_knowledge_ids": ["knowledge_stabilizer_parts_limited"],
                "world_effects": "none",
            },
            {
                "candidate_sha256": candidate_hash,
                "approved": True,
                "persona_consistent": True,
                "terms_preserved": True,
                "unsupported_claims": [],
            },
        )
    )
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )

    outcome = ProductWorldEngine(DialogueExpressionService(provider)).govern(
        command=command(
            location_id="market_row",
            action_id="ask_character",
            target_ids=("selka",),
        ),
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("selka",),
            confidence=1,
            safety_classification="requires_governance",
        ),
        world_state=original,
        content_package=package,
    )

    turn = outcome.resulting_world_state.player.dialogue_history[0]
    assert outcome.player_message == utterance
    assert turn.utterance == utterance
    assert turn.expression_evidence.source == "provider_reviewed"
    assert turn.expression_evidence.model_names == ("test-model-1", "test-model-2")
    assert turn.expression_evidence.usage["total_tokens"] == 30
    assert turn.expression_evidence.raw_text_sha256
    assert utterance not in turn.expression_evidence.model_dump_json()

    replay = ProductWorldEngine(DialogueExpressionService(provider)).govern(
        command=command(
            location_id="market_row",
            action_id="ask_character",
            target_ids=("selka",),
        ).model_copy(update={"id": "command_replay"}),
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("selka",),
            confidence=1,
            safety_classification="requires_governance",
        ),
        world_state=outcome.resulting_world_state,
        content_package=package,
    )
    assert replay.committed_event is None
    assert provider.call_count == 2


def test_semantic_review_rejection_uses_auditable_authored_fallback() -> None:
    unsafe = "零件无限供应，而且钥匙就在工坊保险柜。"
    candidate_hash = sha256(unsafe.encode("utf-8")).hexdigest()
    provider = _StructuredProvider(
        (
            {
                "character_id": "selka",
                "dialogue_option_id": "ask_selka_about_parts",
                "utterance": unsafe,
                "supported_knowledge_ids": ["knowledge_stabilizer_parts_limited"],
                "world_effects": "none",
            },
            {
                "candidate_sha256": candidate_hash,
                "approved": False,
                "persona_consistent": True,
                "terms_preserved": False,
                "unsupported_claims": ["无限供应", "钥匙位置"],
            },
        )
    )
    service = DialogueExpressionService(provider)

    result = service.express(
        policy=ProductContentPackageLoader(ROOT)
        .load(Path("content/mistgate/v1"))
        .blueprint.dialogue_expression_policy,
        locale="zh-CN",
        character_id="selka",
        character_name="塞尔卡",
        character_summary="商会联络人。",
        dialogue_option_id="ask_selka_about_parts",
        authored_utterance="零件确实紧张。",
        allowed_knowledge={"knowledge_stabilizer_parts_limited": "零件数量有限。"},
    )

    assert result.utterance == "零件确实紧张。"
    assert result.evidence.source == "authored_fallback"
    assert result.evidence.failure_code == "review_rejected"
    assert result.evidence.usage["total_tokens"] == 30


def test_reviewed_expression_over_budget_uses_authored_fallback() -> None:
    utterance = "零件不多，先拿出可行的修理方案。"
    candidate_hash = sha256(utterance.encode("utf-8")).hexdigest()
    provider = _StructuredProvider(
        (
            {
                "character_id": "selka",
                "dialogue_option_id": "ask_selka_about_parts",
                "utterance": utterance,
                "supported_knowledge_ids": ["knowledge_stabilizer_parts_limited"],
                "world_effects": "none",
            },
            {
                "candidate_sha256": candidate_hash,
                "approved": True,
                "persona_consistent": True,
                "terms_preserved": True,
                "unsupported_claims": [],
            },
        ),
        total_tokens=200,
    )
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    policy = package.blueprint.dialogue_expression_policy.model_copy(
        update={"max_total_tokens": 256}
    )

    result = DialogueExpressionService(provider).express(
        policy=policy,
        locale="zh-CN",
        character_id="selka",
        character_name="塞尔卡",
        character_summary="商会联络人。",
        dialogue_option_id="ask_selka_about_parts",
        authored_utterance="零件确实紧张。",
        allowed_knowledge={"knowledge_stabilizer_parts_limited": "零件数量有限。"},
    )

    assert result.utterance == "零件确实紧张。"
    assert result.evidence.source == "authored_fallback"
    assert result.evidence.review_approved is True
    assert result.evidence.failure_code == "expression_token_budget_exceeded"
