import logging

from aethelis.utils.redaction import REDACTED, RedactingFilter, redact_data, redact_text


def test_redact_text_covers_common_credential_forms() -> None:
    raw = (
        "Authorization: Bearer abc.def.ghi "
        "api_key=sk-live-example "
        "url=https://user:password@example.test/path"
    )
    result = redact_text(raw)

    assert "abc.def.ghi" not in result
    assert "sk-live-example" not in result
    assert "password@" not in result
    assert REDACTED in result


def test_redact_data_handles_nested_sensitive_fields() -> None:
    raw = {
        "headers": {"authorization": "Bearer secret-token"},
        "openai_api_key": "sk-live-openai",
        "nested": [{"token": "private-token"}, "sk-inline-secret"],
    }
    result = redact_data(raw)

    assert result["headers"]["authorization"] == REDACTED
    assert result["openai_api_key"] == REDACTED
    assert result["nested"][0]["token"] == REDACTED
    assert result["nested"][1] == REDACTED


def test_logging_filter_sanitizes_message_and_arguments() -> None:
    record = logging.LogRecord(
        name="aethelis.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="provider failed with api_key=%s",
        args=("sk-live-provider-secret",),
        exc_info=None,
    )

    assert RedactingFilter().filter(record)
    rendered = record.getMessage()
    assert "sk-live-provider-secret" not in rendered
    assert REDACTED in rendered
