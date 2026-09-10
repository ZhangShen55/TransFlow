import asyncio
import time

import pytest

from transflow.domain.errors import AdmissionRejected
from transflow.domain.work import TranslationUnit
from transflow.inference.fake import FakeInferenceBackend
from transflow.scheduler.fair import FairInferenceScheduler, RequestAdmission


def make_unit(request_id: str, index: int, text: str | None = None) -> TranslationUnit:
    source = text or f"{request_id}-{index}"
    return TranslationUnit(
        request_id=request_id,
        backend_request_id=f"{request_id}:0:{index}",
        text_index=index,
        language_index=0,
        source_text=source,
        target_language_code="en",
        target_language_name="英语",
        prompt=f"translate {source}",
        deadline=time.monotonic() + 10,
    )


@pytest.mark.asyncio
async def test_admission_accepts_64_and_rejects_65th() -> None:
    admission = RequestAdmission(64)
    contexts = [admission.admit() for _ in range(64)]
    for context in contexts:
        await context.__aenter__()

    with pytest.raises(AdmissionRejected):
        async with admission.admit():
            pass

    for context in reversed(contexts):
        await context.__aexit__(None, None, None)
    assert admission.active == 0


@pytest.mark.asyncio
async def test_scheduler_bounds_inflight_work() -> None:
    backend = FakeInferenceBackend(delays={f"r-{index}": 0.01 for index in range(6)})
    scheduler = FairInferenceScheduler(backend, max_inflight=2, max_pending=10)
    await scheduler.start()

    results = await scheduler.submit(
        "r", [make_unit("r", index, f"r-{index}") for index in range(6)]
    )

    assert len(results) == 6
    assert backend.max_active_count == 2
    await scheduler.close()


@pytest.mark.asyncio
async def test_scheduler_rejects_a_full_internal_queue() -> None:
    backend = FakeInferenceBackend(delays={"slow": 10})
    scheduler = FairInferenceScheduler(backend, max_inflight=1, max_pending=1)
    await scheduler.start()
    first = asyncio.create_task(scheduler.submit("one", [make_unit("one", 0, "slow")]))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(scheduler.submit("two", [make_unit("two", 0, "slow")]))
    await asyncio.sleep(0)

    with pytest.raises(AdmissionRejected):
        await scheduler.submit("three", [make_unit("three", 0)])

    first.cancel()
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    await scheduler.close()


@pytest.mark.asyncio
async def test_scheduler_round_robins_pending_requests() -> None:
    backend = FakeInferenceBackend(delays={"one-0": 0.02})
    scheduler = FairInferenceScheduler(backend, max_inflight=1, max_pending=10)
    await scheduler.start()
    one = asyncio.create_task(
        scheduler.submit("one", [make_unit("one", index) for index in range(3)])
    )
    await asyncio.sleep(0)
    two = asyncio.create_task(scheduler.submit("two", [make_unit("two", 0)]))
    await asyncio.gather(one, two)

    call_ids = [unit.request_id for unit in backend.calls]
    assert call_ids.index("two") < 3
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancelling_request_aborts_active_work() -> None:
    backend = FakeInferenceBackend(delays={"slow": 10})
    scheduler = FairInferenceScheduler(backend, max_inflight=1, max_pending=10)
    await scheduler.start()
    submitted = asyncio.create_task(scheduler.submit("cancel", [make_unit("cancel", 0, "slow")]))
    await asyncio.sleep(0.01)

    await scheduler.cancel_request("cancel")

    assert "cancel:0:0" in backend.aborted
    assert (await asyncio.gather(submitted, return_exceptions=True))[0]
    await scheduler.close()
