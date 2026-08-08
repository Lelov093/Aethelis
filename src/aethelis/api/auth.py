from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol
from urllib.parse import urlsplit

import jwt
from jwt import PyJWKClient
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aethelis.config.settings import DEFAULT_ENV_FILE
from aethelis.product.contracts import PrincipalContext, PrincipalStatus
from aethelis.product.errors import ProductAccessDeniedError
from aethelis.product.ports import ProductUnitOfWork


class OIDCSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHELIS_OIDC_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider_id: str = Field(min_length=1, max_length=80)
    issuer: AnyHttpUrl
    audience: str = Field(min_length=1, max_length=255)
    jwks_url: AnyHttpUrl
    algorithms: str = "RS256"

    @field_validator("provider_id")
    @classmethod
    def reject_placeholder_provider(cls, value: str) -> str:
        if value.strip().lower() in {"replace-me", "changeme", "xxx"}:
            raise ValueError("OIDC provider_id is still a placeholder")
        return value

    @field_validator("algorithms")
    @classmethod
    def allow_asymmetric_algorithms(cls, value: str) -> str:
        allowed = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
        selected = {item.strip() for item in value.split(",") if item.strip()}
        if not selected or not selected.issubset(allowed):
            raise ValueError("OIDC algorithms must be an explicit asymmetric allow-list")
        return ",".join(sorted(selected))

    @field_validator("issuer", "jwks_url")
    @classmethod
    def require_secure_oidc_endpoint(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https" and value.host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("OIDC endpoints must use HTTPS outside local development")
        return value


class VerifiedIdentity(Protocol):
    def verify_subject(self, token: str) -> str: ...


class PrincipalContextResolver(Protocol):
    requires_bearer: bool

    def resolve(self, token: str | None) -> PrincipalContext: ...


class ProductAccessSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHELIS_AUTH_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: Literal["local_single_user", "oidc"] = "local_single_user"


class LocalSingleUserSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHELIS_LOCAL_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    principal_id: str = "principal_local_player"
    profile_id: str = "profile_local_player"
    display_name: str = Field(default="雾门旅人", min_length=1, max_length=80)
    locale: str = Field(default="zh-CN", min_length=2, max_length=35)
    allowed_origin: str = "http://localhost:5173"

    @field_validator("allowed_origin")
    @classmethod
    def require_loopback_origin(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("local allowed origin must be an exact loopback HTTP origin")
        return value.rstrip("/")


class OIDCJWKSVerifier:
    def __init__(self, settings: OIDCSettings) -> None:
        self.provider_id = settings.provider_id
        self._issuer = str(settings.issuer).rstrip("/")
        self._audience = settings.audience
        self._algorithms = tuple(settings.algorithms.split(","))
        self._jwks = PyJWKClient(str(settings.jwks_url), cache_jwk_set=True, lifespan=300)

    def verify_subject(self, token: str) -> str:
        signing_key = self._jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=list(self._algorithms),
            audience=self._audience,
            issuer=self._issuer,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
        return str(claims["sub"])


class PrincipalResolver:
    requires_bearer = True

    def __init__(
        self,
        verifier: VerifiedIdentity,
        uow_factory: Callable[[], ProductUnitOfWork],
        *,
        provider_id: str,
    ) -> None:
        self._verifier = verifier
        self._uow_factory = uow_factory
        self._provider_id = provider_id

    def resolve(self, token: str | None) -> PrincipalContext:
        if token is None:
            raise ProductAccessDeniedError(
                "missing_access_token", "Bearer access token is required."
            )
        try:
            subject = self._verifier.verify_subject(token)
        except Exception as exc:
            raise ProductAccessDeniedError(
                "invalid_access_token", "Access token is invalid."
            ) from exc
        with self._uow_factory() as uow:
            principal = uow.identities.get_principal_by_external_subject(self._provider_id, subject)
            if principal is None or principal.status != PrincipalStatus.ACTIVE:
                raise ProductAccessDeniedError(
                    "principal_not_active", "Authenticated principal is not active."
                )
            return PrincipalContext(principal_id=principal.id, roles=principal.roles)


class LocalSingleUserPrincipalResolver:
    requires_bearer = False

    def __init__(
        self,
        uow_factory: Callable[[], ProductUnitOfWork],
        *,
        principal_id: str,
    ) -> None:
        self._uow_factory = uow_factory
        self._principal_id = principal_id

    def resolve(self, _token: str | None) -> PrincipalContext:
        with self._uow_factory() as uow:
            principal = uow.identities.get_principal(self._principal_id)
            if principal is None or principal.status != PrincipalStatus.ACTIVE:
                raise ProductAccessDeniedError(
                    "principal_not_active", "Local player principal is not active."
                )
            return PrincipalContext(principal_id=principal.id, roles=principal.roles)
