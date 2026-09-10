import pytest

from transflow.domain.languages import (
    SUPPORTED_LANGUAGES,
    target_language_name,
    validate_language_codes,
)


def test_all_documented_languages_have_full_names() -> None:
    assert len(SUPPORTED_LANGUAGES) == 38
    assert all(SUPPORTED_LANGUAGES.values())


def test_arabic_code_is_ar() -> None:
    assert target_language_name("ar") == "阿拉伯语"
    with pytest.raises(ValueError, match="不支持"):
        target_language_name("ra")


def test_duplicate_and_unknown_languages_are_rejected() -> None:
    with pytest.raises(ValueError, match="重复"):
        validate_language_codes(["en", "en"])
    with pytest.raises(ValueError, match="不支持"):
        validate_language_codes(["xx"])
