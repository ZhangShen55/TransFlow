from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from transflow.core.logging import get_logger
from transflow.domain.errors import (
    AdmissionRejected,
    BackendFailure,
    BackendUnavailable,
    SchedulerClosed,
    TranslationTimeout,
)
from transflow.schemas.translation import (
    ErrorDetail,
    ErrorResponse,
    TranslateRequest,
    TranslateResponse,
)
from transflow.services.translation import generate_request_id

if TYPE_CHECKING:
    from transflow.core.config import Settings


def _error_response(
    status: int,
    code: str,
    message: str,
    request_id: str,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status, content=body.model_dump())


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _translate_until_disconnect(
    payload: TranslateRequest,
    request: Request,
    request_id: str,
) -> TranslateResponse:
    translation = asyncio.create_task(
        request.app.state.translation_service.translate(payload, request_id=request_id)
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({translation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            translation.cancel()
            await request.app.state.scheduler.cancel_request(request_id)
            await asyncio.gather(translation, return_exceptions=True)
            raise asyncio.CancelledError
        return cast(TranslateResponse, await translation)
    finally:
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/translate",
        summary="翻译文本",
        response_description="完整翻译结果",
        response_model=TranslateResponse,
        responses={
            422: {"model": ErrorResponse, "description": "请求参数校验失败"},
            429: {"model": ErrorResponse, "description": "请求准入容量已耗尽"},
            502: {"model": ErrorResponse, "description": "翻译任务执行失败"},
            503: {"model": ErrorResponse, "description": "翻译后端未就绪"},
            504: {"model": ErrorResponse, "description": "翻译请求处理超时"},
        },
    )
    async def translate(payload: TranslateRequest, request: Request) -> Response:
        request_id = generate_request_id(settings.response.id_prefix)
        started = time.monotonic()
        status = 500
        logger = get_logger().bind(
            request_id=request_id,
            text_items=len(payload.text),
            target_languages=len(payload.language),
        )
        try:
            result = await _translate_until_disconnect(payload, request, request_id)
            status = 200
            logger.info(
                "translation_completed",
                process_time_ms=result.result.process_time_ms,
            )
            return JSONResponse(
                status_code=200,
                content=result.model_dump(),
                headers={"X-Request-Id": request_id},
            )
        except ValueError as exc:
            status = 422
            return _error_response(status, "validation_error", str(exc), request_id)
        except AdmissionRejected as exc:
            status = 429
            return _error_response(status, "capacity_exhausted", str(exc), request_id)
        except BackendUnavailable as exc:
            status = 503
            return _error_response(status, "backend_unavailable", str(exc), request_id)
        except TranslationTimeout as exc:
            status = 504
            return _error_response(status, "translation_timeout", str(exc), request_id)
        except (BackendFailure, SchedulerClosed) as exc:
            status = 502
            return _error_response(status, "translation_failed", str(exc), request_id)
        except asyncio.CancelledError:
            await request.app.state.scheduler.cancel_request(request_id)
            logger.info("translation_cancelled")
            raise
        finally:
            request.app.state.metrics.record_request(
                status=status,
                seconds=time.monotonic() - started,
                text_items=len(payload.text),
                languages=len(payload.language),
            )

    @router.get(
        "/health/live",
        summary="检查服务存活状态",
        response_description="服务存活状态",
    )
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @router.get(
        "/health/ready",
        summary="检查服务就绪状态",
        response_description="服务就绪状态",
        responses={503: {"description": "翻译后端未就绪"}},
    )
    async def ready(request: Request) -> Response:
        if await request.app.state.backend.is_healthy():
            return JSONResponse({"status": "ready"})
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @router.get(settings.metrics.path, include_in_schema=False)
    async def metrics(request: Request) -> Response:
        if not settings.metrics.enabled:
            return Response(status_code=404)
        return Response(
            content=request.app.state.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return router
