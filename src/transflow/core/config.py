# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ServerSettings(FrozenSettings):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)

    @field_validator("workers")
    @classmethod
    def single_worker_only(cls, value: int) -> int:
        if value != 1:
            raise ValueError("workers 必须为 1，以确保调度限制在进程内全局生效")
        return value


class ApiSettings(FrozenSettings):
    max_concurrent_requests: int = Field(default=64, ge=1)
    max_text_items: int = Field(default=120, ge=1)
    max_target_languages: int = Field(default=6, ge=1)
    max_chars_per_text: int = Field(default=4096, ge=1)
    max_total_chars: int = Field(default=131_072, ge=1)
    request_timeout_seconds: float = Field(default=600.0, gt=0)


class SchedulerSettings(FrozenSettings):
    dispatch_chunk_size: int = Field(default=60, ge=1)
    max_inflight_sequences: int = Field(default=512, ge=1)
    max_pending_units: int = Field(default=23_040, ge=1)
    shutdown_grace_seconds: float = Field(default=30.0, ge=0)


class InferenceSettings(FrozenSettings):
    base_urls: tuple[str, ...] = ("http://127.0.0.1:8001/v1",)
    model_name: str = "Hy-MT2-1.8B"
    api_key: str = "local"
    health_timeout_seconds: float = Field(default=2.0, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_model_len: int = Field(default=4096, ge=256)
    batch_size: int = Field(default=32, ge=1)
    batch_workers: int = Field(default=16, ge=1)
    batch_wait_milliseconds: float = Field(default=2.0, ge=0)
    max_connections: int = Field(default=16, ge=1)

    @field_validator("base_urls")
    @classmethod
    def normalize_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("base_urls 必须至少包含一个 URL")
        normalized = tuple(url.rstrip("/") for url in value)
        if any(not url.startswith(("http://", "https://")) for url in normalized):
            raise ValueError("base_urls 中的地址必须使用 http:// 或 https://")
        return normalized


class GenerationSettings(FrozenSettings):
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=-1, ge=-1)
    repetition_penalty: float = Field(default=1.05, gt=0)
    seed: int = 0


class ResponseSettings(FrozenSettings):
    public_model_name: str = "seaCraft-translation-model"
    id_prefix: str = "seaCraft-"


class LoggingSettings(FrozenSettings):
    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")
    include_text: bool = False

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("level 必须是标准 Python 日志级别")
        return level


class MetricsSettings(FrozenSettings):
    enabled: bool = True
    path: str = "/metrics"

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("指标路径必须以 '/' 开头")
        return value


class Settings(FrozenSettings):
    server: ServerSettings = ServerSettings()
    api: ApiSettings = ApiSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    inference: InferenceSettings = InferenceSettings()
    generation: GenerationSettings = GenerationSettings()
    response: ResponseSettings = ResponseSettings()
    logging: LoggingSettings = LoggingSettings()
    metrics: MetricsSettings = MetricsSettings()

    @model_validator(mode="after")
    def validate_relationships(self) -> Settings:
        if self.scheduler.dispatch_chunk_size > self.api.max_text_items:
            raise ValueError("scheduler.dispatch_chunk_size 不能超过 api.max_text_items")
        minimum_pending = (
            self.api.max_concurrent_requests
            * self.scheduler.dispatch_chunk_size
            * self.api.max_target_languages
        )
        if self.scheduler.max_pending_units < minimum_pending:
            raise ValueError("scheduler.max_pending_units 必须能容纳每个已准入请求的一个调度分片")
        if self.generation.max_tokens >= self.inference.max_model_len:
            raise ValueError("generation.max_tokens 必须小于 inference.max_model_len")
        if self.inference.max_connections < self.inference.batch_workers:
            raise ValueError("inference.max_connections 不能小于 inference.batch_workers")
        batch_capacity = self.inference.batch_size * self.inference.batch_workers
        if batch_capacity < self.scheduler.max_inflight_sequences:
            raise ValueError("推理批处理容量不能小于 scheduler.max_inflight_sequences")
        return self


def _parse_environment_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_environment_overrides(
    data: dict[str, Any], environ: Mapping[str, str]
) -> dict[str, Any]:
    result = deepcopy(data)
    prefix = "TRANSFLOW_"
    # 双下划线只允许覆盖一层节和字段，防止环境变量意外创建任意嵌套结构。
    for key, raw_value in environ.items():
        if not key.startswith(prefix) or key == "TRANSFLOW_CONFIG":
            continue
        parts = key.removeprefix(prefix).lower().split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        section_data = result.setdefault(section, {})
        if not isinstance(section_data, dict):
            raise ValueError(f"配置节 {section!r} 不是 TOML 表")
        section_data[field] = _parse_environment_value(raw_value)
    return result


def load_settings(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    effective_environ = os.environ if environ is None else environ
    configured_path: str | Path
    if path is not None:
        configured_path = path
    else:
        configured_path = effective_environ.get("TRANSFLOW_CONFIG") or "config.toml"
    config_path = Path(configured_path)
    with config_path.open("rb") as config_file:
        raw_data = tomllib.load(config_file)
    overridden = _apply_environment_overrides(raw_data, effective_environ)
    return Settings.model_validate(overridden)
