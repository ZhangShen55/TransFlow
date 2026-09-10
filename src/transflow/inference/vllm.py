# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from transflow.core.config import GenerationSettings, InferenceSettings
from transflow.domain.errors import BackendFailure, BackendUnavailable, TranslationTimeout
from transflow.domain.work import GenerationResult, TranslationUnit


class _RetryableBackendUnavailable(Exception):
    pass


@dataclass(slots=True)
class _BatchItem:
    unit: TranslationUnit
    future: asyncio.Future[GenerationResult]


class VLLMInferenceBackend:
    def __init__(
        self,
        settings: InferenceSettings,
        generation: GenerationSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.generation = generation
        self._client = client or httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=settings.max_connections,
            )
        )
        self._owns_client = client is None
        self._next_endpoint = 0
        self._active: dict[str, asyncio.Task[GenerationResult]] = {}
        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False

    @staticmethod
    def _service_root(base_url: str) -> str:
        split = urlsplit(base_url)
        path = split.path.removesuffix("/v1")
        return urlunsplit((split.scheme, split.netloc, path, "", "")).rstrip("/")

    def _choose_base_url(self) -> str:
        base_url = self.settings.base_urls[self._next_endpoint % len(self.settings.base_urls)]
        self._next_endpoint += 1
        return base_url

    def _ensure_workers(self) -> None:
        if self._closed:
            raise BackendUnavailable("vLLM 后端已经关闭")
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._batch_worker())
                for _ in range(self.settings.batch_workers)
            ]

    async def is_healthy(self) -> bool:
        for base_url in self.settings.base_urls:
            try:
                response = await self._client.get(
                    f"{self._service_root(base_url)}/health",
                    timeout=self.settings.health_timeout_seconds,
                )
                if response.is_success:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def generate(self, unit: TranslationUnit) -> asyncio.Task[GenerationResult]:
        self._ensure_workers()
        task = asyncio.create_task(self._enqueue(unit))
        self._active[unit.backend_request_id] = task
        task.add_done_callback(lambda _: self._active.pop(unit.backend_request_id, None))
        return task

    async def _enqueue(self, unit: TranslationUnit) -> GenerationResult:
        future: asyncio.Future[GenerationResult] = asyncio.get_running_loop().create_future()
        await self._queue.put(_BatchItem(unit=unit, future=future))
        return await future

    async def _batch_worker(self) -> None:
        try:
            while True:
                first = await self._queue.get()
                items = [first]
                # 在极短窗口内合并独立会话，减少 HTTP 开销，同时不拼接任何原文。
                wait_seconds = self.settings.batch_wait_milliseconds / 1000
                collect_until = time.monotonic() + wait_seconds
                while len(items) < self.settings.batch_size:
                    remaining = collect_until - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                    items.append(item)
                active_items = [item for item in items if not item.future.cancelled()]
                if not active_items:
                    continue
                try:
                    results = await self._generate_batch_with_retries(
                        [item.unit for item in active_items]
                    )
                except asyncio.CancelledError:
                    for item in active_items:
                        item.future.cancel()
                    raise
                except Exception as exc:
                    for item in active_items:
                        if not item.future.done():
                            item.future.set_exception(exc)
                else:
                    for item, result in zip(active_items, results, strict=True):
                        if not item.future.done():
                            item.future.set_result(result)
        except asyncio.CancelledError:
            raise

    async def _generate_batch_with_retries(
        self, units: list[TranslationUnit]
    ) -> list[GenerationResult]:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            base_url = self._choose_base_url()
            try:
                return await self._generate_batch_once(base_url, units)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
            except _RetryableBackendUnavailable as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
            except httpx.ReadTimeout as exc:
                raise TranslationTimeout("等待 vLLM 响应超时") from exc
        raise BackendUnavailable("无法连接任何 vLLM 端点") from last_error

    async def _generate_batch_once(
        self, base_url: str, units: list[TranslationUnit]
    ) -> list[GenerationResult]:
        remaining = min(unit.deadline for unit in units) - time.monotonic()
        if remaining <= 0:
            raise TranslationTimeout("翻译任务在开始推理前已超过期限")
        timeout = min(remaining, self.settings.request_timeout_seconds)
        batch_id = f"{units[0].request_id}:batch:{secrets.token_hex(8)}"
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": [[{"role": "user", "content": unit.prompt}] for unit in units],
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "top_k": self.generation.top_k,
            "repetition_penalty": self.generation.repetition_penalty,
            "seed": self.generation.seed,
            "request_id": batch_id,
        }
        response = await self._client.post(
            f"{base_url}/chat/completions/batch",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "X-Request-Id": batch_id,
            },
            timeout=timeout,
        )
        if response.status_code in {429, 502, 503, 504}:
            raise _RetryableBackendUnavailable(f"vLLM 不可用，HTTP 状态码：{response.status_code}")
        try:
            response.raise_for_status()
            body = response.json()
            choices = body["choices"]
            if not isinstance(choices, list) or len(choices) != len(units):
                raise ValueError("批响应的结果数量与请求数量不一致")
            # vLLM 可以乱序返回 choice，必须按 index 重排后再完成各翻译单元的 future。
            ordered: list[str | None] = [None] * len(units)
            for choice in choices:
                index = choice["index"]
                content = choice["message"]["content"]
                if not isinstance(index, int) or not 0 <= index < len(units):
                    raise ValueError("批响应包含无效的结果索引")
                if not isinstance(content, str) or ordered[index] is not None:
                    raise ValueError("批响应的结果内容无效或索引重复")
                ordered[index] = content
            if any(content is None for content in ordered):
                raise ValueError("批响应缺少结果项")
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            results = [GenerationResult(text=content or "") for content in ordered]
            # 批请求只返回一组 usage，将其记到首项可避免指标重复累计。
            results[0] = GenerationResult(
                text=results[0].text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return results
        except (httpx.HTTPStatusError, KeyError, TypeError, ValueError) as exc:
            raise BackendFailure("vLLM 返回了无效的批响应") from exc

    async def abort(self, backend_request_id: str) -> None:
        task = self._active.get(backend_request_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        workers = list(self._workers)
        for worker in workers:
            worker.cancel()
        if tasks or workers:
            await asyncio.gather(*tasks, *workers, return_exceptions=True)
        self._workers.clear()
        while not self._queue.empty():
            item = self._queue.get_nowait()
            item.future.cancel()
        if self._owns_client:
            await self._client.aclose()
