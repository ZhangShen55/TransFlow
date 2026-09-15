# ruff: noqa: RUF001

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transflow.core.config import ApiSettings
from transflow.domain.languages import normalize_language_code, validate_language_codes


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranslateRequest(ApiModel):
    model_config = ConfigDict(extra="forbid", title="翻译请求")

    text: list[str] = Field(min_length=1, title="待翻译文本数组")
    language: list[str] = Field(min_length=1, title="目标语言代码数组")

    @field_validator("language", mode="before")
    @classmethod
    def normalize_languages(cls, value: object) -> object:
        if isinstance(value, list | tuple):
            return [
                normalize_language_code(item) if isinstance(item, str) else item
                for item in value
            ]
        return value

    @field_validator("language")
    @classmethod
    def validate_languages(cls, value: list[str]) -> list[str]:
        validate_language_codes(value)
        return value


class TranslationContent(ApiModel):
    model_config = ConfigDict(extra="forbid", title="单语言翻译结果")

    content: list[str] = Field(title="译文数组")
    language: str = Field(title="目标语言代码")


class TranslationResult(ApiModel):
    model_config = ConfigDict(extra="forbid", title="翻译结果")

    contents: list[TranslationContent] = Field(title="各目标语言结果")
    finished_time: int = Field(title="完成时间 Unix 秒")
    process_time_ms: int = Field(ge=0, title="处理耗时毫秒")
    finished_reason: Literal["finished"] = Field(default="finished", title="完成原因")


class TranslateResponse(ApiModel):
    model_config = ConfigDict(extra="forbid", title="翻译响应")

    model: str = Field(title="公开模型名称")
    id: str = Field(title="请求标识")
    result: TranslationResult = Field(title="翻译结果")


class ErrorDetail(ApiModel):
    model_config = ConfigDict(extra="forbid", title="错误详情")

    code: str = Field(title="错误代码")
    message: str = Field(title="错误说明")
    request_id: str | None = Field(default=None, title="请求标识")


class ErrorResponse(ApiModel):
    model_config = ConfigDict(extra="forbid", title="错误响应")

    error: ErrorDetail = Field(title="错误详情")


def validate_request_limits(request: TranslateRequest, settings: ApiSettings) -> None:
    if len(request.text) > settings.max_text_items:
        raise ValueError(f"text 包含的条目超过 {settings.max_text_items} 条")
    if len(request.language) > settings.max_target_languages:
        raise ValueError(f"language 包含的条目超过 {settings.max_target_languages} 条")
    oversized = [
        index
        for index, value in enumerate(request.text)
        if len(value) > settings.max_chars_per_text
    ]
    if oversized:
        raise ValueError(
            f"以下索引处的 text 条目超过 {settings.max_chars_per_text} 个字符："
            + ", ".join(str(index) for index in oversized)
        )
    if sum(len(value) for value in request.text) > settings.max_total_chars:
        raise ValueError(f"text 超过 {settings.max_total_chars} 个字符的总量上限")
