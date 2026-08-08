from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path

import uvicorn
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aethelis.api.app import create_app
from aethelis.api.auth import (
    LocalSingleUserPrincipalResolver,
    LocalSingleUserSettings,
    OIDCJWKSVerifier,
    OIDCSettings,
    PrincipalResolver,
    ProductAccessSettings,
)
from aethelis.config.settings import DEFAULT_ENV_FILE
from aethelis.db.command_repository import SQLAlchemyCommandRepository
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.db.product_repository import sqlalchemy_product_uow_factory
from aethelis.product.command_service import CommandApplicationService
from aethelis.product.local_mode import bootstrap_local_single_user
from aethelis.product.projections import ProjectionService
from aethelis.product.services import ProductApplicationService


class APIRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHELIS_API_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


def build_app():  # type: ignore[no-untyped-def]
    engine = create_db_engine(load_database_settings())
    uow_factory = sqlalchemy_product_uow_factory(engine)
    access = ProductAccessSettings()
    allowed_origins: tuple[str, ...] = ()
    if access.mode == "local_single_user":
        local = LocalSingleUserSettings()
        bootstrap_local_single_user(
            uow_factory=uow_factory,
            repository_root=Path(__file__).resolve().parents[3],
            principal_id=local.principal_id,
            profile_id=local.profile_id,
            display_name=local.display_name,
            locale=local.locale,
        )
        resolver = LocalSingleUserPrincipalResolver(
            uow_factory,
            principal_id=local.principal_id,
        )
        allowed_origins = (local.allowed_origin,)
    else:
        oidc = OIDCSettings()
        verifier = OIDCJWKSVerifier(oidc)
        resolver = PrincipalResolver(
            verifier,
            uow_factory,
            provider_id=oidc.provider_id,
        )
    commands = CommandApplicationService(uow_factory, SQLAlchemyCommandRepository(engine))
    product = ProductApplicationService(uow_factory)
    return create_app(
        command_service=commands,
        product_service=product,
        principal_resolver=resolver,
        projection_service=ProjectionService(uow_factory),
        allowed_origins=allowed_origins,
    )


def main() -> None:
    settings = APIRuntimeSettings()
    access = ProductAccessSettings()
    validate_runtime_binding(settings.host, access.mode)
    uvicorn.run(build_app(), host=settings.host, port=settings.port)


def validate_runtime_binding(host: str, auth_mode: str) -> None:
    if auth_mode != "local_single_user":
        return
    normalized = host.strip().lower()
    if normalized == "localhost":
        return
    try:
        if ip_address(normalized).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(
        "local_single_user mode requires a loopback API host; use 127.0.0.1, ::1, or localhost"
    )
