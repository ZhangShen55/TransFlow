# ruff: noqa: RUF003

from __future__ import annotations

import asyncio
import secrets
import time

from transflow.core.config import Settings
from transflow.domain.errors import BackendUnavailable, TranslationTimeout
from transflow.domain.languages import target_language_name
from transflow.domain.work import TranslationUnit
from transflow.inference.prompts import build_translation_prompt
from transflow.scheduler.fair import FairInferenceScheduler, RequestAdmission
from transflow.schemas.translation import (
    TranslateRequest,
    TranslateResponse,
    TranslationContent,
    TranslationResult,
    validate_request_limits,
)


def generate_request_id(prefix: str) -> str:
    return f"{prefix}{time.time_ns():016x}{secrets.token_hex(8)}"


class TranslationService:
    def __init__(
        self,
        settings: Settings,
        scheduler: FairInferenceScheduler,
        admission: RequestAdmission,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler
        self.admission = admission

    async def translate(
        self, request: TranslateRequest, *, request_id: str | None = None
    ) -> TranslateResponse:
        validate_request_limits(request, self.settings.api)
        effective_id = request_id or generate_request_id(self.settings.response.id_prefix)
        started = time.monotonic()
        deadline = started + self.settings.api.request_timeout_seconds
        if not await self.scheduler.backend.is_healthy():
            raise BackendUnavailable("翻译后端尚未就绪")

        async with self.admission.admit():
            # 矩阵坐标固定为 [目标语言索引][文本索引]，空字符串天然保持原位置。
            matrix = [[""] * len(request.text) for _ in request.language]
            try:
                async with asyncio.timeout_at(deadline):
                    await self._fill_matrix(request, effective_id, deadline, matrix)
            except TimeoutError as exc:
                await self.scheduler.cancel_request(effective_id)
                raise TranslationTimeout("翻译请求已超过处理期限") from exc

        finished = time.time()
        process_time_ms = max(0, round((time.monotonic() - started) * 1000))
        return TranslateResponse(
            model=self.settings.response.public_model_name,
            id=effective_id,
            result=TranslationResult(
                contents=[
                    TranslationContent(content=matrix[index], language=language)
                    for index, language in enumerate(request.language)
                ],
                finished_time=int(finished),
                process_time_ms=process_time_ms,
            ),
        )

    async def _fill_matrix(
        self,
        request: TranslateRequest,
        request_id: str,
        deadline: float,
        matrix: list[list[str]],
    ) -> None:
        chunk_size = self.settings.scheduler.dispatch_chunk_size
        # 分片按文本位置推进；每个非空位置再展开为各目标语言的独立推理单元。
        for chunk_start in range(0, len(request.text), chunk_size):
            units: list[TranslationUnit] = []
            for text_index in range(chunk_start, min(chunk_start + chunk_size, len(request.text))):
                source_text = request.text[text_index]
                if source_text == "":
                    continue
                for language_index, language_code in enumerate(request.language):
                    language_name = target_language_name(language_code)
                    units.append(
                        TranslationUnit(
                            request_id=request_id,
                            backend_request_id=(f"{request_id}:{language_index}:{text_index}"),
                            text_index=text_index,
                            language_index=language_index,
                            source_text=source_text,
                            target_language_code=language_code,
                            target_language_name=language_name,
                            prompt=build_translation_prompt(source_text, language_name),
                            deadline=deadline,
                        )
                    )
            results = await self.scheduler.submit(request_id, units)
            for unit, result in zip(units, results, strict=True):
                matrix[unit.language_index][unit.text_index] = result.text
