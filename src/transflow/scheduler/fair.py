# ruff: noqa: RUF003

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Protocol

from transflow.domain.errors import AdmissionRejected, SchedulerClosed
from transflow.domain.work import GenerationResult, TranslationUnit
from transflow.inference.base import InferenceBackend


class RequestAdmission:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._active = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._closed or self._active >= self.capacity:
                raise AdmissionRejected("请求准入容量已经耗尽")
            self._active += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active -= 1

    async def close(self) -> None:
        async with self._lock:
            self._closed = True


@dataclass(slots=True)
class _PendingUnit:
    unit: TranslationUnit
    future: asyncio.Future[GenerationResult]
    enqueued_at: float


@dataclass(slots=True)
class _ActiveUnit:
    item: _PendingUnit
    task: asyncio.Task[None]


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    pending: int
    active: int
    requests: int


class SchedulerObserver(Protocol):
    def update_scheduler(self, snapshot: SchedulerSnapshot) -> None: ...

    def record_queue_wait(self, seconds: float) -> None: ...

    def record_inference(
        self,
        seconds: float,
        outcome: str,
        result: GenerationResult | None,
    ) -> None: ...


class NullSchedulerObserver:
    def update_scheduler(self, snapshot: SchedulerSnapshot) -> None:
        pass

    def record_queue_wait(self, seconds: float) -> None:
        pass

    def record_inference(
        self,
        seconds: float,
        outcome: str,
        result: GenerationResult | None,
    ) -> None:
        pass


class FairInferenceScheduler:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        max_inflight: int,
        max_pending: int,
        observer: SchedulerObserver | None = None,
    ) -> None:
        self.backend = backend
        self.max_inflight = max_inflight
        self.max_pending = max_pending
        self.observer = observer or NullSchedulerObserver()
        self._queues: dict[str, deque[_PendingUnit]] = {}
        self._order: deque[str] = deque()
        self._active: dict[str, _ActiveUnit] = {}
        self._pending_count = 0
        self._closed = False
        self._condition = asyncio.Condition()
        self._dispatcher: asyncio.Task[None] | None = None
        self._idle = asyncio.Event()
        self._idle.set()

    async def start(self) -> None:
        async with self._condition:
            if self._dispatcher is None:
                self._dispatcher = asyncio.create_task(self._dispatch_loop())

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            pending=self._pending_count,
            active=len(self._active),
            requests=len(self._queues),
        )

    async def submit(
        self, request_id: str, units: Sequence[TranslationUnit]
    ) -> list[GenerationResult]:
        if not units:
            return []
        loop = asyncio.get_running_loop()
        enqueued_at = time.monotonic()
        items = [_PendingUnit(unit, loop.create_future(), enqueued_at) for unit in units]
        async with self._condition:
            if self._closed:
                raise SchedulerClosed("调度器已经关闭")
            if request_id in self._queues:
                raise RuntimeError(f"请求 {request_id!r} 已经存在待处理分片")
            if self._pending_count + len(items) > self.max_pending:
                raise AdmissionRejected("推理单元队列已满")
            self._queues[request_id] = deque(items)
            self._order.append(request_id)
            self._pending_count += len(items)
            self.observer.update_scheduler(self.snapshot())
            self._condition.notify_all()
        try:
            return list(await asyncio.gather(*(item.future for item in items)))
        except BaseException:
            await self.cancel_request(request_id)
            raise

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: self._closed
                        or (bool(self._order) and len(self._active) < self.max_inflight)
                    )
                    if self._closed:
                        return
                    # 每次只从一个请求取一个单元并将请求放回队尾，实现请求间轮询公平。
                    request_id = self._order.popleft()
                    queue = self._queues[request_id]
                    item = queue.popleft()
                    self._pending_count -= 1
                    if queue:
                        self._order.append(request_id)
                    else:
                        del self._queues[request_id]
                    task = asyncio.create_task(self._execute(item))
                    self._active[item.unit.backend_request_id] = _ActiveUnit(item, task)
                    self._idle.clear()
                    self.observer.update_scheduler(self.snapshot())
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise

    async def _execute(self, item: _PendingUnit) -> None:
        started = time.monotonic()
        self.observer.record_queue_wait(started - item.enqueued_at)
        result: GenerationResult | None = None
        outcome = "success"
        try:
            result = await self.backend.generate(item.unit)
            if not item.future.done():
                item.future.set_result(result)
        except asyncio.CancelledError:
            outcome = "cancelled"
            if not item.future.done():
                item.future.cancel()
            raise
        except Exception as exc:
            outcome = "failure"
            if not item.future.done():
                item.future.set_exception(exc)
        finally:
            self.observer.record_inference(
                time.monotonic() - started,
                outcome,
                result,
            )
            async with self._condition:
                self._active.pop(item.unit.backend_request_id, None)
                if not self._active:
                    self._idle.set()
                self.observer.update_scheduler(self.snapshot())
                self._condition.notify_all()

    async def cancel_request(self, request_id: str) -> None:
        async with self._condition:
            queue = self._queues.pop(request_id, deque())
            if queue:
                self._pending_count -= len(queue)
                for item in queue:
                    item.future.cancel()
            self._order = deque(candidate for candidate in self._order if candidate != request_id)
            active = [
                active
                for active in self._active.values()
                if active.item.unit.request_id == request_id
            ]
            for active_unit in active:
                active_unit.task.cancel()
            self.observer.update_scheduler(self.snapshot())
            self._condition.notify_all()
        # 网络中止操作放在条件锁之外，避免慢后端阻塞其他请求的调度与取消。
        if active:
            await asyncio.gather(
                *(self.backend.abort(item.item.unit.backend_request_id) for item in active),
                return_exceptions=True,
            )
            await asyncio.gather(*(item.task for item in active), return_exceptions=True)

    async def close(self, grace_seconds: float = 0.0) -> None:
        async with self._condition:
            self._closed = True
            request_ids = set(self._queues)
            request_ids.update(item.item.unit.request_id for item in self._active.values())
            self._condition.notify_all()
        if grace_seconds > 0 and self._active:
            with suppress(TimeoutError):
                async with asyncio.timeout(grace_seconds):
                    await self._idle.wait()
        for request_id in request_ids:
            await self.cancel_request(request_id)
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
            self._dispatcher = None
