from collections.abc import Awaitable
from typing import Protocol

from transflow.domain.work import GenerationResult, TranslationUnit


class InferenceBackend(Protocol):
    async def is_healthy(self) -> bool: ...

    def generate(self, unit: TranslationUnit) -> Awaitable[GenerationResult]: ...

    async def abort(self, backend_request_id: str) -> None: ...

    async def close(self) -> None: ...
