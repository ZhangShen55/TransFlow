# ruff: noqa: RUF001, RUF003

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from transflow.api.routes import create_router
from transflow.core.config import Settings, load_settings
from transflow.core.logging import configure_logging
from transflow.inference.base import InferenceBackend
from transflow.inference.vllm import VLLMInferenceBackend
from transflow.observability.metrics import Metrics
from transflow.scheduler.fair import FairInferenceScheduler, RequestAdmission
from transflow.schemas.translation import ErrorDetail, ErrorResponse
from transflow.services.translation import TranslationService, generate_request_id


def _format_validation_error(exc: RequestValidationError) -> str:
    # 只提取字段位置和错误类型，避免把 Pydantic 错误中的原始文本回显给调用方。
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"] if part != "body") or "请求体"
        error_type = error["type"]
        context = error.get("ctx") or {}
        if error_type == "value_error" and context.get("error") is not None:
            detail = str(context["error"])
        elif error_type == "missing":
            detail = "缺少必填字段"
        elif error_type == "too_short":
            detail = f"至少需要 {context.get('min_length', 1)} 项"
        elif error_type == "too_long":
            detail = f"最多允许 {context.get('max_length')} 项"
        elif error_type == "list_type":
            detail = "必须是数组"
        elif error_type == "string_type":
            detail = "必须是字符串"
        elif error_type == "extra_forbidden":
            detail = "不允许额外字段"
        else:
            detail = "值无效"
        messages.append(f"{location}: {detail}")
    return "请求参数校验失败：" + "；".join(messages)


def create_app(
    *,
    settings: Settings | None = None,
    backend: InferenceBackend | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    effective_settings = settings or load_settings(config_path)
    configure_logging(effective_settings.logging)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 后端、调度器和准入器必须属于同一应用生命周期，确保容量限制不会跨实例失效。
        effective_backend = backend or VLLMInferenceBackend(
            effective_settings.inference,
            effective_settings.generation,
        )
        metrics = Metrics()
        scheduler = FairInferenceScheduler(
            effective_backend,
            max_inflight=effective_settings.scheduler.max_inflight_sequences,
            max_pending=effective_settings.scheduler.max_pending_units,
            observer=metrics,
        )
        admission = RequestAdmission(effective_settings.api.max_concurrent_requests)
        await scheduler.start()
        app.state.settings = effective_settings
        app.state.backend = effective_backend
        app.state.metrics = metrics
        app.state.scheduler = scheduler
        app.state.admission = admission
        app.state.translation_service = TranslationService(
            effective_settings,
            scheduler,
            admission,
        )
        try:
            yield
        finally:
            await admission.close()
            await scheduler.close(effective_settings.scheduler.shutdown_grace_seconds)
            await effective_backend.close()

    app = FastAPI(
        title="TransFlow",
        description="基于 Hy-MT2 的多语言翻译服务",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(create_router(effective_settings))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        request_id = request.headers.get("X-Request-Id") or generate_request_id(
            effective_settings.response.id_prefix
        )
        error = ErrorResponse(
            error=ErrorDetail(
                code="validation_error",
                message=_format_validation_error(_exc),
                request_id=request_id,
            )
        )
        if hasattr(request.app.state, "metrics"):
            request.app.state.metrics.record_request(
                status=422,
                seconds=0.0,
                text_items=0,
                languages=0,
            )
        return JSONResponse(status_code=422, content=error.model_dump())

    return app


app = create_app()
