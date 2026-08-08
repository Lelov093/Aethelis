"""local timelines and named saves

Revision ID: 20260802_0007
Revises: 20260801_0006
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260802_0007"
down_revision: str | None = "20260801_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "product_world_instances",
        sa.Column(
            "name",
            sa.String(120),
            nullable=False,
            server_default="未命名时间线",
        ),
    )
    op.alter_column("product_world_instances", "name", server_default=None)
    op.add_column(
        "product_world_instances",
        sa.Column("forked_from_world_instance_id", sa.String(120)),
    )
    op.add_column(
        "product_world_instances",
        sa.Column("forked_from_save_point_id", sa.String(120)),
    )
    op.add_column(
        "product_world_instances",
        sa.Column("forked_from_snapshot_id", sa.String(120)),
    )
    op.create_check_constraint(
        "ck_world_instance_fork_provenance",
        "product_world_instances",
        "(forked_from_world_instance_id is null and forked_from_save_point_id is null "
        "and forked_from_snapshot_id is null) or "
        "(forked_from_world_instance_id is not null and forked_from_save_point_id is not null "
        "and forked_from_snapshot_id is not null)",
    )
    op.create_index(
        "ix_product_world_instances_fork_parent",
        "product_world_instances",
        ["forked_from_world_instance_id"],
    )
    op.add_column(
        "product_save_points",
        sa.Column("name", sa.String(120)),
    )


def downgrade() -> None:
    op.drop_column("product_save_points", "name")
    op.drop_index(
        "ix_product_world_instances_fork_parent",
        table_name="product_world_instances",
    )
    op.drop_constraint(
        "ck_world_instance_fork_provenance",
        "product_world_instances",
        type_="check",
    )
    op.drop_column("product_world_instances", "forked_from_snapshot_id")
    op.drop_column("product_world_instances", "forked_from_save_point_id")
    op.drop_column("product_world_instances", "forked_from_world_instance_id")
    op.drop_column("product_world_instances", "name")
