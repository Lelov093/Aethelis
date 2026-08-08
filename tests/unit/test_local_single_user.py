import pytest
from pydantic import ValidationError

from aethelis.api.auth import LocalSingleUserSettings, ProductAccessSettings
from aethelis.api.bootstrap import validate_runtime_binding


def test_local_mode_is_default_and_rejects_non_loopback_binding() -> None:
    settings = ProductAccessSettings(_env_file=None)

    assert settings.mode == "local_single_user"
    validate_runtime_binding("127.0.0.1", settings.mode)
    validate_runtime_binding("localhost", settings.mode)
    validate_runtime_binding("::1", settings.mode)
    with pytest.raises(ValueError, match="loopback"):
        validate_runtime_binding("0.0.0.0", settings.mode)
    with pytest.raises(ValueError, match="loopback"):
        validate_runtime_binding("192.168.1.8", settings.mode)


def test_local_origin_must_be_an_exact_loopback_origin() -> None:
    settings = LocalSingleUserSettings(
        _env_file=None,
        allowed_origin="http://localhost:5173/",
    )

    assert settings.allowed_origin == "http://localhost:5173"
    for unsafe in (
        "https://example.com",
        "http://0.0.0.0:5173",
        "http://localhost:5173/callback",
        "http://localhost:5173?redirect=evil",
    ):
        with pytest.raises(ValidationError):
            LocalSingleUserSettings(_env_file=None, allowed_origin=unsafe)
