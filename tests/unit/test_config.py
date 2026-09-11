from pathlib import Path

import pytest
from pydantic import ValidationError

from transflow.core.config import Settings, load_settings

CONFIG_PATH = Path(__file__).parents[2] / "config.toml"


def test_project_config_loads() -> None:
    settings = load_settings(CONFIG_PATH, environ={})

    assert settings.api.max_text_items == 120
    assert settings.api.max_target_languages == 12
    assert settings.scheduler.dispatch_chunk_size == 60
    assert settings.scheduler.max_inflight_sequences == 512
    assert settings.scheduler.max_pending_units == 46_080
    assert settings.inference.batch_size == 32
    assert settings.inference.batch_workers == 16
    assert settings.inference.max_connections == 16
    assert settings.inference.base_urls == ("http://127.0.0.1:8001/v1",)


def test_environment_override_is_typed() -> None:
    settings = load_settings(
        CONFIG_PATH,
        environ={
            "TRANSFLOW_INFERENCE__BASE_URLS": '["http://vllm:8000/v1"]',
            "TRANSFLOW_LOGGING__LEVEL": "debug",
        },
    )

    assert settings.inference.base_urls == ("http://vllm:8000/v1",)
    assert settings.logging.level == "DEBUG"


def test_invalid_cross_field_limits_fail() -> None:
    with pytest.raises(ValidationError, match="dispatch_chunk_size"):
        Settings.model_validate(
            {
                "api": {"max_text_items": 10},
                "scheduler": {"dispatch_chunk_size": 11},
            }
        )


def test_pending_queue_must_cover_one_chunk_per_request() -> None:
    with pytest.raises(ValidationError, match="max_pending_units"):
        Settings.model_validate({"scheduler": {"max_pending_units": 100}})


def test_extra_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Settings.model_validate({"api": {"unknown_limit": 1}})
