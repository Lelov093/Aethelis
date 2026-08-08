from sqlalchemy import CheckConstraint, UniqueConstraint

import aethelis.db.product_models  # noqa: F401
from aethelis.db.models import Base
from aethelis.db.product_models import PRODUCT_TABLE_NAMES


def test_product_metadata_registers_dedicated_tables() -> None:
    assert set(PRODUCT_TABLE_NAMES).issubset(Base.metadata.tables)
    assert "runs" not in PRODUCT_TABLE_NAMES
    assert "world_state_snapshots" not in PRODUCT_TABLE_NAMES


def test_product_tables_expose_version_and_identity_constraints() -> None:
    snapshot = Base.metadata.tables["product_world_snapshots"]
    principal = Base.metadata.tables["product_principals"]
    instances = Base.metadata.tables["product_world_instances"]
    saves = Base.metadata.tables["product_save_points"]

    assert any(isinstance(item, UniqueConstraint) for item in snapshot.constraints)
    assert any(isinstance(item, CheckConstraint) for item in snapshot.constraints)
    assert any(isinstance(item, UniqueConstraint) for item in principal.constraints)
    assert snapshot.c.world_instance_id.foreign_keys
    assert instances.c.name.nullable is False
    assert instances.c.forked_from_save_point_id is not None
    assert saves.c.name is not None
