import asyncio

import pytest

from transflow.core.config import ApiSettings, SchedulerSettings, Settings
from transflow.domain.errors import BackendFailure
from transflow.inference.fake import FakeInferenceBackend
from transflow.scheduler.fair import FairInferenceScheduler, RequestAdmission
from transflow.schemas.translation import TranslateRequest
from transflow.services.translation import TranslationService


async def make_service(
    backend: FakeInferenceBackend,
    *,
    settings: Settings | None = None,
) -> tuple[TranslationService, FairInferenceScheduler]:
    effective_settings = settings or Settings()
    scheduler = FairInferenceScheduler(
        backend,
        max_inflight=effective_settings.scheduler.max_inflight_sequences,
        max_pending=effective_settings.scheduler.max_pending_units,
    )
    await scheduler.start()
    return (
        TranslationService(
            effective_settings,
            scheduler,
            RequestAdmission(effective_settings.api.max_concurrent_requests),
        ),
        scheduler,
    )


@pytest.mark.asyncio
async def test_translation_preserves_language_and_text_order() -> None:
    backend = FakeInferenceBackend(
        translations={
            ("慢", "en"): "slow",
            ("快", "en"): "fast",
            ("慢", "ar"): "بطيء",
            ("快", "ar"): "سريع",
        },
        delays={"慢": 0.02},
    )
    service, scheduler = await make_service(backend)

    response = await service.translate(
        TranslateRequest(text=["", "慢", "快", ""], language=["en", "ar"])
    )

    assert [entry.language for entry in response.result.contents] == ["en", "ar"]
    assert response.result.contents[0].content == ["", "slow", "fast", ""]
    assert response.result.contents[1].content == ["", "بطيء", "سريع", ""]
    assert len(backend.calls) == 4
    await scheduler.close()


@pytest.mark.asyncio
async def test_all_empty_request_does_not_generate() -> None:
    backend = FakeInferenceBackend()
    service, scheduler = await make_service(backend)

    response = await service.translate(TranslateRequest(text=["", ""], language=["zh", "en"]))

    assert all(entry.content == ["", ""] for entry in response.result.contents)
    assert backend.calls == []
    await scheduler.close()


@pytest.mark.asyncio
async def test_120_entries_are_submitted_as_two_chunks() -> None:
    backend = FakeInferenceBackend()
    settings = Settings(
        api=ApiSettings(max_target_languages=1),
        scheduler=SchedulerSettings(
            dispatch_chunk_size=60,
            max_inflight_sequences=128,
            max_pending_units=3840,
        ),
    )
    service, scheduler = await make_service(backend, settings=settings)
    batch_sizes: list[int] = []
    original_submit = scheduler.submit

    async def recording_submit(request_id: str, units: list[object]) -> list[object]:
        batch_sizes.append(len(units))
        return await original_submit(request_id, units)  # type: ignore[arg-type, return-value]

    scheduler.submit = recording_submit  # type: ignore[method-assign, assignment]
    response = await service.translate(
        TranslateRequest(text=[f"line-{index}" for index in range(120)], language=["en"])
    )

    assert batch_sizes == [60, 60]
    assert len(response.result.contents[0].content) == 120
    await scheduler.close()


@pytest.mark.asyncio
async def test_failure_is_atomic_and_cancels_remaining_work() -> None:
    backend = FakeInferenceBackend(fail_texts={"bad"}, delays={"slow": 10})
    service, scheduler = await make_service(backend)

    with pytest.raises(BackendFailure):
        await service.translate(
            TranslateRequest(text=["bad", "slow"], language=["en"]),
            request_id="atomic",
        )

    await asyncio.sleep(0)
    assert scheduler.snapshot().pending == 0
    assert scheduler.snapshot().active == 0
    await scheduler.close()


@pytest.mark.asyncio
async def test_same_language_and_duplicate_texts_are_generated_independently() -> None:
    backend = FakeInferenceBackend()
    service, scheduler = await make_service(backend)

    response = await service.translate(TranslateRequest(text=["你好", "你好"], language=["zh"]))

    assert response.result.contents[0].content == ["[zh] 你好", "[zh] 你好"]
    assert len(backend.calls) == 2
    await scheduler.close()
