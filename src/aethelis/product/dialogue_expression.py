from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from pydantic import Field

from aethelis.llm.base import LLMProvider, StructuredLLMResult
from aethelis.product.content_contracts import ProductDialogueExpressionPolicy
from aethelis.providers import ProviderError
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.world import DialogueExpressionEvidence


class DialogueExpressionDraft(AethelisModel):
    character_id: Identifier
    dialogue_option_id: Identifier
    utterance: str = Field(min_length=1, max_length=500)
    supported_knowledge_ids: tuple[Identifier, ...] = ()
    world_effects: Literal["none"]


class DialogueExpressionReview(AethelisModel):
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved: bool
    persona_consistent: bool
    terms_preserved: bool
    unsupported_claims: tuple[str, ...] = Field(default=(), max_length=8)


@dataclass(frozen=True)
class DialogueExpression:
    utterance: str
    evidence: DialogueExpressionEvidence


class DialogueExpressionService:
    """Generate and review surface dialogue without granting it world authority."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def express(
        self,
        *,
        policy: ProductDialogueExpressionPolicy,
        locale: str,
        character_id: str,
        character_name: str,
        character_summary: str,
        dialogue_option_id: str,
        authored_utterance: str,
        allowed_knowledge: dict[str, str],
        required_terms: tuple[str, ...] = (),
        conversation_context: tuple[dict[str, str], ...] = (),
    ) -> DialogueExpression:
        if not policy.enabled:
            return _authored(authored_utterance)

        try:
            draft_result = self._provider.generate_structured(
                _draft_prompt(
                    locale=locale,
                    character_id=character_id,
                    character_name=character_name,
                    character_summary=character_summary,
                    dialogue_option_id=dialogue_option_id,
                    authored_utterance=authored_utterance,
                    allowed_knowledge=allowed_knowledge,
                    required_terms=required_terms,
                    conversation_context=conversation_context,
                    max_characters=policy.max_utterance_characters,
                ),
                DialogueExpressionDraft,
                max_tokens=420,
                temperature=0.45,
            )
        except ProviderError as exc:
            return _fallback(authored_utterance, "draft_provider_failure", exc.provider)
        except Exception:
            return _fallback(authored_utterance, "draft_unexpected_failure")

        failure = _validate_draft(
            draft_result,
            character_id=character_id,
            dialogue_option_id=dialogue_option_id,
            allowed_knowledge_ids=tuple(allowed_knowledge),
            max_characters=policy.max_utterance_characters,
        )
        if failure is not None or draft_result.data is None:
            return _fallback_from_results(
                authored_utterance,
                failure or "draft_invalid",
                draft_result,
            )

        draft = draft_result.data
        candidate_hash = sha256(draft.utterance.encode("utf-8")).hexdigest()
        try:
            review_result = self._provider.generate_structured(
                _review_prompt(
                    locale=locale,
                    character_name=character_name,
                    character_summary=character_summary,
                    authored_utterance=authored_utterance,
                    allowed_knowledge=allowed_knowledge,
                    required_terms=required_terms,
                    conversation_context=conversation_context,
                    candidate=draft,
                    candidate_hash=candidate_hash,
                ),
                DialogueExpressionReview,
                max_tokens=320,
                temperature=0.0,
            )
        except ProviderError as exc:
            return _fallback_from_results(
                authored_utterance,
                "review_provider_failure",
                draft_result,
                provider_name=exc.provider,
            )
        except Exception:
            return _fallback_from_results(
                authored_utterance,
                "review_unexpected_failure",
                draft_result,
            )

        review_failure = _validate_review(review_result, candidate_hash)
        if review_failure is not None:
            review = review_result.data
            return _fallback_from_results(
                authored_utterance,
                review_failure,
                draft_result,
                review_result,
                review_approved=review.approved if review is not None else None,
                unsupported_claim_count=(
                    len(review.unsupported_claims) if review is not None else 0
                ),
            )
        budget_failure = _validate_budget(policy, draft_result, review_result)
        if budget_failure is not None:
            return _fallback_from_results(
                authored_utterance,
                budget_failure,
                draft_result,
                review_result,
                review_approved=True,
            )
        return DialogueExpression(
            utterance=draft.utterance,
            evidence=_evidence(
                "provider_reviewed",
                draft_result,
                review_result,
                review_approved=True,
            ),
        )


def resolve_dialogue_expression(
    service: DialogueExpressionService | None,
    *,
    policy: ProductDialogueExpressionPolicy,
    locale: str,
    character_id: str,
    character_name: str,
    character_summary: str,
    dialogue_option_id: str,
    authored_utterance: str,
    allowed_knowledge: dict[str, str],
    required_terms: tuple[str, ...] = (),
    conversation_context: tuple[dict[str, str], ...] = (),
) -> DialogueExpression:
    if service is None:
        return _authored(authored_utterance)
    return service.express(
        policy=policy,
        locale=locale,
        character_id=character_id,
        character_name=character_name,
        character_summary=character_summary,
        dialogue_option_id=dialogue_option_id,
        authored_utterance=authored_utterance,
        allowed_knowledge=allowed_knowledge,
        required_terms=required_terms,
        conversation_context=conversation_context,
    )


def _draft_prompt(
    *,
    locale: str,
    character_id: str,
    character_name: str,
    character_summary: str,
    dialogue_option_id: str,
    authored_utterance: str,
    allowed_knowledge: dict[str, str],
    required_terms: tuple[str, ...],
    conversation_context: tuple[dict[str, str], ...],
    max_characters: int,
) -> str:
    context = {
        "locale": locale,
        "character": {
            "id": character_id,
            "name": character_name,
            "public_summary": character_summary,
        },
        "dialogue_option_id": dialogue_option_id,
        "authoritative_authored_utterance": authored_utterance,
        "author_text_is_fully_authorized": True,
        "allowed_knowledge": allowed_knowledge,
        "required_terms": required_terms,
        "recent_visible_conversation": conversation_context,
        "max_utterance_characters": max_characters,
    }
    return (
        "Create one natural character utterance as surface expression only. "
        "Paraphrase the authoritative utterance without adding facts, promises, costs, "
        "permissions, resource quantities, locations, people, or world effects. "
        "Use recent visible conversation only for conversational continuity, never as new "
        "world truth. Use only allowed knowledge for factual claims. Echo the exact "
        "character_id, dialogue_option_id, and "
        "all allowed knowledge IDs. Set world_effects to none.\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def _review_prompt(
    *,
    locale: str,
    character_name: str,
    character_summary: str,
    authored_utterance: str,
    allowed_knowledge: dict[str, str],
    required_terms: tuple[str, ...],
    conversation_context: tuple[dict[str, str], ...],
    candidate: DialogueExpressionDraft,
    candidate_hash: str,
) -> str:
    context = {
        "locale": locale,
        "character_name": character_name,
        "public_character_summary": character_summary,
        "authoritative_authored_utterance": authored_utterance,
        "allowed_knowledge": allowed_knowledge,
        "required_terms": required_terms,
        "recent_visible_conversation": conversation_context,
        "candidate": candidate.model_dump(mode="json"),
        "candidate_sha256": candidate_hash,
    }
    return (
        "Review the candidate only against the authoritative authored utterance. Every "
        "meaning already present in that authored utterance is explicitly authorized, even "
        "when it is not repeated in allowed_knowledge. Approve a faithful paraphrase when "
        "it preserves persona and required terms, adds no new meaning, and changes no world "
        "effect. Do not reject merely because the authored utterance contains detail beyond "
        "the compact knowledge record. "
        "Echo candidate_sha256 exactly. List every unsupported claim; when uncertain reject.\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def _validate_draft(
    result: StructuredLLMResult[DialogueExpressionDraft],
    *,
    character_id: str,
    dialogue_option_id: str,
    allowed_knowledge_ids: tuple[str, ...],
    max_characters: int,
) -> str | None:
    if not result.success or result.data is None:
        return "draft_structured_output_invalid"
    draft = result.data
    if draft.character_id != character_id or draft.dialogue_option_id != dialogue_option_id:
        return "draft_identity_mismatch"
    if tuple(draft.supported_knowledge_ids) != allowed_knowledge_ids:
        return "draft_knowledge_scope_mismatch"
    if len(draft.utterance) > max_characters:
        return "draft_too_long"
    return None


def _validate_review(
    result: StructuredLLMResult[DialogueExpressionReview],
    candidate_hash: str,
) -> str | None:
    if not result.success or result.data is None:
        return "review_structured_output_invalid"
    review = result.data
    if review.candidate_sha256 != candidate_hash:
        return "review_candidate_mismatch"
    if (
        not review.approved
        or not review.persona_consistent
        or not review.terms_preserved
        or review.unsupported_claims
    ):
        return "review_rejected"
    return None


def _validate_budget(
    policy: ProductDialogueExpressionPolicy,
    *results: StructuredLLMResult,
) -> str | None:
    if sum(result.latency_ms for result in results) > policy.max_total_latency_ms:
        return "expression_latency_budget_exceeded"
    total_tokens = sum(result.usage.get("total_tokens", 0) for result in results)
    if total_tokens > policy.max_total_tokens:
        return "expression_token_budget_exceeded"
    return None


def _authored(utterance: str) -> DialogueExpression:
    return DialogueExpression(
        utterance=utterance,
        evidence=DialogueExpressionEvidence(source="authored"),
    )


def _fallback(
    utterance: str,
    failure_code: str,
    provider_name: str | None = None,
) -> DialogueExpression:
    return DialogueExpression(
        utterance=utterance,
        evidence=DialogueExpressionEvidence(
            source="authored_fallback",
            provider_name=provider_name,
            failure_code=failure_code,
        ),
    )


def _fallback_from_results(
    utterance: str,
    failure_code: str,
    *results: StructuredLLMResult,
    provider_name: str | None = None,
    review_approved: bool | None = None,
    unsupported_claim_count: int = 0,
) -> DialogueExpression:
    return DialogueExpression(
        utterance=utterance,
        evidence=_evidence(
            "authored_fallback",
            *results,
            failure_code=failure_code,
            provider_name=provider_name,
            review_approved=review_approved,
            unsupported_claim_count=unsupported_claim_count,
        ),
    )


def _evidence(
    source: Literal["provider_reviewed", "authored_fallback"],
    *results: StructuredLLMResult,
    failure_code: str | None = None,
    provider_name: str | None = None,
    review_approved: bool | None = None,
    unsupported_claim_count: int = 0,
) -> DialogueExpressionEvidence:
    usage: dict[str, int] = {}
    for result in results:
        for key, value in result.usage.items():
            usage[key] = usage.get(key, 0) + value
    return DialogueExpressionEvidence(
        source=source,
        provider_name=provider_name or (results[0].provider_name if results else None),
        model_names=tuple(result.model_name for result in results),
        raw_text_sha256=tuple(result.raw_text_sha256 for result in results),
        latency_ms=sum(result.latency_ms for result in results),
        usage=usage,
        review_approved=review_approved,
        unsupported_claim_count=unsupported_claim_count,
        failure_code=failure_code,
    )
