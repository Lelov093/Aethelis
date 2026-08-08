from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.contracts import (
    PlayerProfile,
    PrincipalContext,
    PrincipalRole,
    PrincipalStatus,
    ProductPrincipal,
)
from aethelis.product.errors import ProductCompatibilityError, ProductConflictError
from aethelis.product.ports import ProductUnitOfWork
from aethelis.product.services import ProductApplicationService


def bootstrap_local_single_user(
    *,
    uow_factory,
    repository_root: Path,
    principal_id: str,
    profile_id: str,
    display_name: str,
    locale: str,
    now: datetime | None = None,
) -> tuple[PrincipalContext, PlayerProfile]:
    created_at = now or datetime.now(UTC)
    roles = (PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR)
    with uow_factory() as uow:
        principal = _ensure_principal(uow, principal_id, roles, created_at)
        profile = _ensure_profile(
            uow,
            profile_id=profile_id,
            principal_id=principal.id,
            display_name=display_name,
            locale=locale,
            created_at=created_at,
        )
        uow.commit()

    context = PrincipalContext(principal_id=principal.id, roles=principal.roles)
    package = ProductContentPackageLoader(repository_root).load(Path("content/mistgate/v1"))
    with uow_factory() as uow:
        version = uow.catalog.get_content_version(package.blueprint.content_version_id)
        stored_package = uow.catalog.get_content_package(package.blueprint.content_version_id)
    if version is None and stored_package is None:
        ProductApplicationService(uow_factory, clock=lambda: created_at).publish_content_package(
            principal=context,
            package=package,
        )
    elif version is None or stored_package is None:
        raise ProductConflictError(
            "local_content_bootstrap_incomplete",
            "Local Mistgate content bootstrap is incomplete.",
        )
    elif stored_package.content_hash != version.content_hash:
        raise ProductCompatibilityError(
            "local_content_hash_mismatch",
            "Local Mistgate content package does not match its published version.",
        )
    return context, profile


def _ensure_principal(
    uow: ProductUnitOfWork,
    principal_id: str,
    roles: tuple[PrincipalRole, ...],
    created_at: datetime,
) -> ProductPrincipal:
    principal = uow.identities.get_principal(principal_id)
    if principal is None:
        principal = ProductPrincipal(
            id=principal_id,
            identity_provider="local_single_user",
            external_subject="local_player",
            roles=roles,
            status=PrincipalStatus.ACTIVE,
            created_at=created_at,
        )
        uow.identities.add_principal(principal)
        return principal
    if (
        principal.identity_provider != "local_single_user"
        or principal.external_subject != "local_player"
        or principal.status != PrincipalStatus.ACTIVE
        or not set(roles).issubset(principal.roles)
    ):
        raise ProductCompatibilityError(
            "local_principal_mismatch",
            "Existing local principal is incompatible with local-single-user mode.",
        )
    return principal


def _ensure_profile(
    uow: ProductUnitOfWork,
    *,
    profile_id: str,
    principal_id: str,
    display_name: str,
    locale: str,
    created_at: datetime,
) -> PlayerProfile:
    profile = uow.identities.get_profile(profile_id)
    if profile is None:
        profile = PlayerProfile(
            id=profile_id,
            principal_id=principal_id,
            display_name=display_name,
            locale=locale,
            created_at=created_at,
            updated_at=created_at,
        )
        uow.identities.add_profile(profile)
        return profile
    if profile.principal_id != principal_id:
        raise ProductCompatibilityError(
            "local_profile_mismatch",
            "Existing local player profile belongs to another principal.",
        )
    return profile
