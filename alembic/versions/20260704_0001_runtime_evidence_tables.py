"""runtime evidence tables

Revision ID: 20260704_0001
Revises:
Create Date: 2026-07-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260704_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("seed_id", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("algorithm_mode", sa.String(length=80), nullable=False),
        sa.Column("provider_called", sa.Boolean(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("db_persisted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "run_steps",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("scenario_id", sa.String(length=120), nullable=False),
        sa.Column("agent_id", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("provider_called", sa.Boolean(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("structured_validation_passed", sa.Boolean(), nullable=False),
        sa.Column("state_diff_applied", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "provider_call_records",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("provider_name", sa.String(length=120), nullable=False),
        sa.Column("model_name", sa.String(length=240), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("usage_json", _jsonb(), nullable=False),
        sa.Column("attempts_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "llm_input_records",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column(
            "provider_call_id",
            sa.String(length=120),
            sa.ForeignKey("provider_call_records.id", ondelete="CASCADE"),
        ),
        sa.Column("prompt_sha256", sa.String(length=64), nullable=False),
        sa.Column("redaction_status", sa.String(length=80), nullable=False),
        sa.Column("schema_name", sa.String(length=120), nullable=False),
    )
    op.create_table(
        "llm_output_records",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column(
            "provider_call_id",
            sa.String(length=120),
            sa.ForeignKey("provider_call_records.id", ondelete="CASCADE"),
        ),
        sa.Column("raw_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("raw_output_saved", sa.Boolean(), nullable=False),
        sa.Column("redaction_status", sa.String(length=80), nullable=False),
        sa.Column("structured_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "structured_output_validations",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column(
            "provider_call_id",
            sa.String(length=120),
            sa.ForeignKey("provider_call_records.id", ondelete="CASCADE"),
        ),
        sa.Column("schema_name", sa.String(length=120), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("json_parse_error", sa.Text(), nullable=True),
        sa.Column("validation_error", sa.Text(), nullable=True),
    )
    op.create_table(
        "action_proposals",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("proposal_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "event_candidates",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column(
            "action_proposal_id", sa.String(length=120), sa.ForeignKey("action_proposals.id")
        ),
        sa.Column("candidate_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "verification_results",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column(
            "event_candidate_id", sa.String(length=120), sa.ForeignKey("event_candidates.id")
        ),
        sa.Column("decision", sa.String(length=40), nullable=False),
        sa.Column("verification_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "verifier_check_results",
        sa.Column("id", sa.String(length=160), primary_key=True),
        sa.Column(
            "verification_result_id",
            sa.String(length=120),
            sa.ForeignKey("verification_results.id", ondelete="CASCADE"),
        ),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
    )
    op.create_table(
        "committed_events",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column(
            "event_candidate_id", sa.String(length=120), sa.ForeignKey("event_candidates.id")
        ),
        sa.Column(
            "verification_result_id",
            sa.String(length=120),
            sa.ForeignKey("verification_results.id"),
        ),
        sa.Column("committed_event_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "state_diffs",
        sa.Column("id", sa.String(length=120), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column(
            "committed_event_id", sa.String(length=120), sa.ForeignKey("committed_events.id")
        ),
        sa.Column(
            "event_candidate_id", sa.String(length=120), sa.ForeignKey("event_candidates.id")
        ),
        sa.Column("state_diff_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "state_patches",
        sa.Column("id", sa.String(length=160), primary_key=True),
        sa.Column(
            "state_diff_id",
            sa.String(length=120),
            sa.ForeignKey("state_diffs.id", ondelete="CASCADE"),
        ),
        sa.Column("patch_index", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=40), nullable=False),
        sa.Column("target_type", sa.String(length=80), nullable=False),
        sa.Column("target_id", sa.String(length=120), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("patch_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "world_state_snapshots",
        sa.Column("id", sa.String(length=160), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("snapshot_kind", sa.String(length=40), nullable=False),
        sa.Column("world_state_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "algorithm_decisions",
        sa.Column("id", sa.String(length=180), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("mechanism_id", sa.String(length=120), nullable=False),
        sa.Column("model_family", sa.String(length=160), nullable=False),
        sa.Column("formula", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=120), nullable=False),
        sa.Column("runtime_object_type", sa.String(length=80), nullable=False),
        sa.Column("runtime_object_id", sa.String(length=160), nullable=False),
        sa.Column("input_features_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "algorithm_score_breakdowns",
        sa.Column("id", sa.String(length=220), primary_key=True),
        sa.Column(
            "algorithm_decision_id",
            sa.String(length=180),
            sa.ForeignKey("algorithm_decisions.id", ondelete="CASCADE"),
        ),
        sa.Column("score_name", sa.String(length=120), nullable=False),
        sa.Column("score_value", sa.Float(), nullable=True),
        sa.Column("detail_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "trace_events",
        sa.Column("id", sa.String(length=180), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("event_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "evidence_artifacts",
        sa.Column("id", sa.String(length=180), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("artifact_type", sa.String(length=120), nullable=False),
        sa.Column("redaction_status", sa.String(length=120), nullable=False),
        sa.Column("artifact_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "metric_results",
        sa.Column("id", sa.String(length=180), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column("metric_name", sa.String(length=120), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("metric_json", _jsonb(), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "metric_results",
        "evidence_artifacts",
        "trace_events",
        "algorithm_score_breakdowns",
        "algorithm_decisions",
        "world_state_snapshots",
        "state_patches",
        "state_diffs",
        "committed_events",
        "verifier_check_results",
        "verification_results",
        "event_candidates",
        "action_proposals",
        "structured_output_validations",
        "llm_output_records",
        "llm_input_records",
        "provider_call_records",
        "run_steps",
        "runs",
    ):
        op.drop_table(table)
