import asyncio
from collections.abc import Mapping

from transflow.domain.errors import BackendFailure
from transflow.domain.work import GenerationResult, TranslationUnit


class FakeInferenceBackend:
    def __init__(
        self,
        *,
        translations: Mapping[tuple[str, str], str] | None = None,
        delays: Mapping[str, float] | None = None,
        fail_texts: set[str] | None = None,
        healthy: bool = True,
    ) -> None:
        self.translations = dict(translations or {})
        self.delays = dict(delays or {})
        self.fail_texts = set(fail_texts or set())
        self.healthy = healthy
        self.calls: list[TranslationUnit] = []
        self.aborted: set[str] = set()
        self._active: dict[str, asyncio.Task[GenerationResult]] = {}
        self.active_count = 0
        self.max_active_count = 0
        self.closed = False
        self.started = asyncio.Event()

    async def is_healthy(self) -> bool:
        return self.healthy

    def generate(self, unit: TranslationUnit) -> asyncio.Task[GenerationResult]:
        task = asyncio.create_task(self._generate(unit))
        self._active[unit.backend_request_id] = task
        task.add_done_callback(lambda _: self._active.pop(unit.backend_request_id, None))
        return task

    async def _generate(self, unit: TranslationUnit) -> GenerationResult:
        self.calls.append(unit)
        self.started.set()
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)
        try:
            await asyncio.sleep(self.delays.get(unit.source_text, 0.0))
            if unit.source_text in self.fail_texts:
                raise BackendFailure(f"模拟后端处理 {unit.source_text!r} 时失败")
            text = self.translations.get(
                (unit.source_text, unit.target_language_code),
                f"[{unit.target_language_code}] {unit.source_text}",
            )
            return GenerationResult(text=text, prompt_tokens=1, completion_tokens=1)
        finally:
            self.active_count -= 1

    async def abort(self, backend_request_id: str) -> None:
        self.aborted.add(backend_request_id)
        task = self._active.get(backend_request_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.closed = True
