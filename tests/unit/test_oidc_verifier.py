from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from aethelis.api.auth import OIDCJWKSVerifier, OIDCSettings


class StaticJWKClient:
    def __init__(self, key) -> None:
        self._key = key

    def get_signing_key_from_jwt(self, _token: str):
        return SimpleNamespace(key=self._key)


def verifier(public_key) -> OIDCJWKSVerifier:
    value = OIDCJWKSVerifier.__new__(OIDCJWKSVerifier)
    value.provider_id = "test_oidc"
    value._issuer = "https://identity.example.test"
    value._audience = "aethelis-api"
    value._algorithms = ("RS256",)
    value._jwks = StaticJWKClient(public_key)
    return value


def test_oidc_verifier_requires_signature_issuer_audience_and_subject() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(UTC)
    claims = {
        "sub": "external-user-1",
        "iss": "https://identity.example.test",
        "aud": "aethelis-api",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    token = jwt.encode(claims, private_key, algorithm="RS256")

    assert verifier(private_key.public_key()).verify_subject(token) == "external-user-1"

    wrong_audience = jwt.encode({**claims, "aud": "another-api"}, private_key, algorithm="RS256")
    with pytest.raises(jwt.InvalidAudienceError):
        verifier(private_key.public_key()).verify_subject(wrong_audience)


def test_oidc_settings_reject_non_http_urls() -> None:
    with pytest.raises(ValueError):
        OIDCSettings(
            provider_id="test",
            issuer="not-a-url",
            audience="aethelis-api",
            jwks_url="also-not-a-url",
        )
