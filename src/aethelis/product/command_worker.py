from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Event, Thread
from typing import Protocol

from aethelis.llm.base import LLMProvider
from aethelis.product.command_contracts import (
    CommandInputMode,
    ParsedPlayerIntent,
    PlayerCommand,
    PlayerCommandStatus,
)
from aethelis.schemas.world import DialogueActKind

PRODUCT_ACTION_CATALOG = {
    "investigate_area": "调查或搜索当前地点; investigate/search the current place",
    "inspect_resource": "查看一个可见物件; inspect one visible resource",
    "move_to_location": "前往一个已知地点; travel to a known reachable location",
    "ask_character": "询问、交谈或向人物打听; ask/speak with one visible character",
    "negotiate_resource": "协商资源交换; negotiate a governed resource exchange",
    "repair_regulator": "尝试维修调节器; attempt the regulator repair",
    "break_commitment": "明确放弃承诺; explicitly abandon the active commitment",
    "request_calibration_key": "请求校准钥匙; request the calibration key",
    "validate_gate_lens": "检查并验证透镜; inspect and validate the gate lens",
    "stabilize_regulator": "完成调节器校准; complete regulator calibration",
    "wait_for_world_response": "明确等待世界回应; deliberately wait, never use as fallback",
    "advance_world": "advance bounded world time and allow character action",
    "ask_world": "consult the player-visible world narrative or attempt a world action",
}
UNSAFE_CLASSIFICATIONS = {"unsafe", "blocked", "disallowed", "harmful"}
TARGET_REQUIRED_ACTIONS = {
    "inspect_resource",
    "move_to_location",
    "ask_character",
    "negotiate_resource",
    "repair_regulator",
    "break_commitment",
    "request_calibration_key",
    "validate_gate_lens",
    "stabilize_regulator",
}
CANONICAL_MISSING_FIELDS = {"action", "target", "location", "intent"}
GOVERNANCE_BYPASS_PHRASES = (
    "绕过规则",
    "绕过治理",
    "直接修改世界",
    "直接把世界",
    "bypass governance",
    "bypass the rules",
    "rewrite world truth",
)
HIGH_PRECISION_ACTION_TERMS = (
    ("break_commitment", ("放弃承诺", "违背承诺", "break the commitment")),
    ("request_calibration_key", ("校准钥匙", "calibration key")),
    (
        "validate_gate_lens",
        ("验证透镜", "检查透镜", "查看透镜", "闸门透镜", "validate the lens"),
    ),
    ("stabilize_regulator", ("完成校准", "稳定调节器", "stabilize the regulator")),
    ("repair_regulator", ("维修调节器", "修理调节器", "repair the regulator")),
    ("negotiate_resource", ("协商", "交换", "交易", "negotiate", "trade")),
    ("ask_character", ("问问", "询问", "打听", "交谈", "谈谈", "聊聊", "ask", "speak", "talk")),
    ("investigate_area", ("调查", "搜索", "搜查", "investigate", "search")),
    ("inspect_resource", ("查看", "检查", "观察", "inspect", "examine")),
    ("move_to_location", ("前往", "去往", "抵达", "travel", "go to")),
    ("wait_for_world_response", ("等待", "等一会", "等消息", "wait")),
)


class LeaseCommandRepository(Protocol):
    def claim_next(self, *, worker_id: str, now: datetime, lease_duration: timedelta): ...

    def finish_attempt(self, **kwargs) -> PlayerCommand: ...

    def heartbeat(self, **kwargs) -> None: ...


class IntentParser(Protocol):
    def parse(self, command: PlayerCommand) -> ParsedPlayerIntent: ...


class StructuredIntentParser:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    def parse(self, command: PlayerCommand) -> ParsedPlayerIntent:
        if command.input_mode == CommandInputMode.CONTEXTUAL_ACTION:
            return ParsedPlayerIntent(
                normalized_action=command.action_id or "invalid",
                actor_id=command.actor_id,
                target_ids=command.target_ids,
                confidence=1.0,
                safety_classification="requires_governance",
            )
        normalized_text = (command.text or "").casefold()
        if any(phrase in normalized_text for phrase in GOVERNANCE_BYPASS_PHRASES):
            raise IntentParserRejected("intent asks to bypass governed world truth")
        conversational = _parse_conversation_intent(command, normalized_text)
        if conversational is not None:
            return conversational
        bounded = _parse_high_precision_intent(command, normalized_text)
        if bounded is not None:
            return bounded
        if self._provider is None:
            raise IntentParserUnavailable("natural-language provider is not configured")
        prompt = json.dumps(
            {
                "task": (
                    "Normalize the bounded player intent. Select exactly one action_id from "
                    "allowed_actions, or use missing_fields to request clarification. Do not "
                    "decide or mutate world truth. Do not invent targets or outcomes."
                ),
                "text": command.text,
                "actor_id": command.actor_id,
                "visible_target_ids": command.target_ids,
                "visible_targets": command.target_hints,
                "location_id": command.location_id,
                "locale": command.locale,
                "allowed_actions": PRODUCT_ACTION_CATALOG,
                "routing_examples": {
                    "问问罗文发生了什么 / speak with Rowan": "ask_character",
                    "调查广场 / search this area": "investigate_area",
                    "前往中央档案馆 / go to the archive": "move_to_location",
                    "查看零件 / inspect the parts": "inspect_resource",
                    "等一会看看城市如何回应 / wait for news": "wait_for_world_response",
                },
                "output_rules": {
                    "normalized_action": "one allowed action id",
                    "actor_id": "must equal the supplied actor_id",
                    "target_ids": "zero or more supplied visible_target_ids only",
                    "missing_fields": "only action, target, location, or intent",
                    "safety_classification": "requires_governance or unsafe",
                    "important": (
                        "wait_for_world_response is valid only when the player explicitly "
                        "asks to wait; never use it for unknown or conversational input"
                    ),
                    "dialogue_act": (
                        "for ask_character use greeting, question, claim, or request; "
                        "for ask_world use world_observation or world_action; otherwise null"
                    ),
                    "claim_text": "only for a claim; preserve the player's asserted content",
                },
            },
            ensure_ascii=False,
        )
        result = self._provider.generate_structured(
            prompt, ParsedPlayerIntent, max_tokens=512, temperature=0.0
        )
        if not result.success or result.data is None:
            raise IntentParserInvalidOutput("provider output did not match ParsedPlayerIntent")
        parsed = result.data
        if parsed.actor_id != command.actor_id:
            raise IntentParserInvalidOutput("provider changed the player actor")
        if parsed.normalized_action not in PRODUCT_ACTION_CATALOG:
            raise IntentParserRejected(
                f"unsupported normalized action: {parsed.normalized_action}"
            )
        if not set(parsed.target_ids).issubset(command.target_ids):
            raise IntentParserInvalidOutput("provider selected a target outside the visible scene")
        if parsed.safety_classification.lower() in UNSAFE_CLASSIFICATIONS:
            raise IntentParserRejected("intent is outside the safe playable action boundary")
        parsed = parsed.model_copy(
            update={
                "missing_fields": tuple(
                    field for field in parsed.missing_fields if field in CANONICAL_MISSING_FIELDS
                )
            }
        )
        if parsed.normalized_action in TARGET_REQUIRED_ACTIONS and len(parsed.target_ids) != 1:
            parsed = parsed.model_copy(
                update={"missing_fields": tuple(dict.fromkeys((*parsed.missing_fields, "target")))}
            )
        if parsed.confidence < 0.35 and not parsed.missing_fields:
            parsed = parsed.model_copy(update={"missing_fields": ("intent",)})
        return parsed.model_copy(
            update={
                "provider_name": result.provider_name,
                "model_name": result.model_name,
                "raw_text_sha256": result.raw_text_sha256,
            }
        )


class IntentParserUnavailable(RuntimeError):
    pass


class IntentParserInvalidOutput(RuntimeError):
    pass


class IntentParserRejected(RuntimeError):
    pass


def _parse_high_precision_intent(
    command: PlayerCommand, normalized_text: str
) -> ParsedPlayerIntent | None:
    action = next(
        (
            action_id
            for action_id, terms in HIGH_PRECISION_ACTION_TERMS
            if any(term in normalized_text for term in terms)
        ),
        None,
    )
    if action is None:
        return None
    matched_targets = tuple(
        target_id
        for target_id in command.target_ids
        if _target_matches(
            normalized_text,
            target_id,
            command.target_hints.get(target_id, target_id),
        )
    )
    missing_fields: tuple[str, ...] = ()
    if action in TARGET_REQUIRED_ACTIONS and len(matched_targets) != 1:
        missing_fields = ("target",)
        matched_targets = ()
    return ParsedPlayerIntent(
        normalized_action=action,
        actor_id=command.actor_id,
        target_ids=matched_targets,
        constraints=("deterministic_high_precision",),
        confidence=0.98 if not missing_fields else 0.72,
        missing_fields=missing_fields,
        safety_classification="requires_governance",
        raw_text_sha256=sha256((command.text or "").encode("utf-8")).hexdigest(),
    )


def _parse_conversation_intent(
    command: PlayerCommand, normalized_text: str
) -> ParsedPlayerIntent | None:
    raw_text = (command.text or "").strip()
    source_hash = sha256(raw_text.encode("utf-8")).hexdigest()
    if "world_narrative" in command.target_ids:
        concrete_action = next(
            (
                action_id
                for action_id, terms in HIGH_PRECISION_ACTION_TERMS
                if action_id != "ask_character" and any(term in normalized_text for term in terms)
            ),
            None,
        )
        if concrete_action is not None:
            return None
        action_terms = (
            "我想",
            "我要",
            "尝试",
            "打开",
            "移动",
            "检查",
            "搜索",
            "try",
            "open",
        )
        act = (
            DialogueActKind.WORLD_ACTION
            if any(term in normalized_text for term in action_terms)
            else DialogueActKind.WORLD_OBSERVATION
        )
        return ParsedPlayerIntent(
            normalized_action="ask_world",
            actor_id=command.actor_id,
            target_ids=(),
            constraints=("visible_world_narrative",),
            confidence=0.99,
            safety_classification="requires_governance",
            dialogue_act=act,
            raw_text_sha256=source_hash,
        )
    if len(command.target_ids) != 1:
        return None
    target_id = command.target_ids[0]
    greeting_terms = ("你好", "您好", "嗨", "早上好", "晚上好", "hello", "hi", "hey")
    claim_terms = (
        "我听说",
        "听说",
        "据说",
        "我知道",
        "我发现",
        "告诉你",
        "i heard",
        "i know",
    )
    if any(_dialogue_phrase_matches(normalized_text, term) for term in greeting_terms):
        act = DialogueActKind.GREETING
        claim_text = None
    elif any(_dialogue_phrase_matches(normalized_text, term) for term in claim_terms):
        act = DialogueActKind.CLAIM
        claim_text = raw_text
    else:
        return None
    return ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id=command.actor_id,
        target_ids=(target_id,),
        constraints=("focused_dialogue_target",),
        confidence=0.96,
        safety_classification="requires_governance",
        dialogue_act=act,
        claim_text=claim_text,
        raw_text_sha256=source_hash,
    )


def _dialogue_phrase_matches(text: str, phrase: str) -> bool:
    if phrase.isascii():
        return re.search(rf"\b{re.escape(phrase)}\b", text) is not None
    return phrase in text


def _target_matches(text: str, target_id: str, label: str) -> bool:
    aliases = {target_id.casefold(), label.casefold()}
    aliases.update(
        token.casefold()
        for token in re.split(r"[\s/·|,，()（）\-]+", label)
        if len(token.strip()) >= 2
    )
    return any(alias and alias in text for alias in aliases)


class CommandWorker:
    def __init__(
        self,
        repository: LeaseCommandRepository,
        parser: IntentParser,
        *,
        worker_id: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        lease_duration: timedelta = timedelta(seconds=90),
    ) -> None:
        self._repository = repository
        self._parser = parser
        self._worker_id = worker_id
        self._clock = clock
        self._lease_duration = lease_duration

    def run_once(self) -> PlayerCommand | None:
        claimed = self._repository.claim_next(
            worker_id=self._worker_id,
            now=self._clock(),
            lease_duration=self._lease_duration,
        )
        if claimed is None:
            return None
        command, _execution = claimed
        stop_heartbeat = Event()
        heartbeat = Thread(
            target=self._heartbeat_loop,
            args=(command.id, stop_heartbeat),
            daemon=True,
        )
        heartbeat.start()
        try:
            intent = self._parser.parse(command)
            status = (
                PlayerCommandStatus.NEEDS_CLARIFICATION
                if intent.missing_fields
                else PlayerCommandStatus.READY_FOR_GOVERNANCE
            )
            return self._repository.finish_attempt(
                command_id=command.id,
                worker_id=self._worker_id,
                now=self._clock(),
                status=status,
                parsed_intent=intent,
            )
        except IntentParserUnavailable as exc:
            return self._repository.finish_attempt(
                command_id=command.id,
                worker_id=self._worker_id,
                now=self._clock(),
                status=PlayerCommandStatus.FAILED,
                error_code="intent_provider_unavailable",
                error_message=str(exc),
                retryable=True,
            )
        except IntentParserInvalidOutput as exc:
            return self._repository.finish_attempt(
                command_id=command.id,
                worker_id=self._worker_id,
                now=self._clock(),
                status=PlayerCommandStatus.FAILED,
                error_code="intent_invalid_output",
                error_message=str(exc),
                retryable=False,
            )
        except IntentParserRejected as exc:
            return self._repository.finish_attempt(
                command_id=command.id,
                worker_id=self._worker_id,
                now=self._clock(),
                status=PlayerCommandStatus.REJECTED,
                error_code="intent_outside_play_boundary",
                error_message=str(exc),
                retryable=False,
            )
        except Exception:
            return self._repository.finish_attempt(
                command_id=command.id,
                worker_id=self._worker_id,
                now=self._clock(),
                status=PlayerCommandStatus.FAILED,
                error_code="intent_provider_failure",
                error_message="Intent provider failed before producing a validated result.",
                retryable=True,
            )
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=1)

    def _heartbeat_loop(self, command_id: str, stop: Event) -> None:
        interval = max(self._lease_duration.total_seconds() / 3, 1)
        while not stop.wait(interval):
            try:
                self._repository.heartbeat(
                    command_id=command_id,
                    worker_id=self._worker_id,
                    now=self._clock(),
                    lease_duration=self._lease_duration,
                )
            except LookupError:
                return
