import pytest
from pydantic import ValidationError

from transflow.core.config import ApiSettings
from transflow.schemas.translation import TranslateRequest, validate_request_limits


def test_one_and_120_text_entries_are_valid() -> None:
    single = TranslateRequest(text=["hello"], language=["zh"])
    maximum = TranslateRequest(text=["hello"] * 120, language=["en", "ar"])

    validate_request_limits(single, ApiSettings())
    validate_request_limits(maximum, ApiSettings())


def test_dynamic_text_and_language_limits() -> None:
    with pytest.raises(ValueError, match="超过 2 条"):
        validate_request_limits(
            TranslateRequest(text=["a", "b", "c"], language=["en"]),
            ApiSettings(max_text_items=2),
        )
    with pytest.raises(ValueError, match="language 包含的条目超过 1 条"):
        validate_request_limits(
            TranslateRequest(text=["a"], language=["en", "fr"]),
            ApiSettings(max_target_languages=1),
        )


def test_schema_rejects_duplicate_and_unsupported_languages() -> None:
    with pytest.raises(ValidationError, match="重复"):
        TranslateRequest(text=["hello"], language=["en", "en"])
    with pytest.raises(ValidationError, match="不支持"):
        TranslateRequest(text=["hello"], language=["ra"])


def test_character_limits_do_not_trim_input() -> None:
    request = TranslateRequest(text=["   "], language=["en"])
    validate_request_limits(request, ApiSettings(max_chars_per_text=3))
    assert request.text == ["   "]
