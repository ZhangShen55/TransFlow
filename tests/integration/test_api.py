# ruff: noqa: RUF001

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from transflow.core.config import ApiSettings, SchedulerSettings, Settings
from transflow.inference.fake import FakeInferenceBackend
from transflow.main import create_app


def settings_for_test(
    *,
    max_concurrent: int = 64,
    timeout: float = 2.0,
) -> Settings:
    api = ApiSettings(
        max_concurrent_requests=max_concurrent,
        max_text_items=120,
        max_target_languages=6,
        request_timeout_seconds=timeout,
    )
    scheduler = SchedulerSettings(
        dispatch_chunk_size=60,
        max_inflight_sequences=16,
        max_pending_units=max_concurrent * 60 * 6,
        shutdown_grace_seconds=0,
    )
    return Settings(api=api, scheduler=scheduler)


@asynccontextmanager
async def api_client(
    backend: FakeInferenceBackend,
    settings: Settings | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings=settings or settings_for_test(), backend=backend)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.mark.asyncio
async def test_translate_contract_and_empty_positions() -> None:
    backend = FakeInferenceBackend(translations={("你好", "en"): "Hello", ("你好", "ar"): "مرحبًا"})
    async with api_client(backend) as client:
        response = await client.post(
            "/translate",
            json={"text": ["", "你好", ""], "language": ["en", "ar"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "seaCraft-translation-model"
    assert body["id"].startswith("seaCraft-")
    assert body["result"]["contents"] == [
        {"content": ["", "Hello", ""], "language": "en"},
        {"content": ["", "مرحبًا", ""], "language": "ar"},
    ]
    assert body["result"]["finished_reason"] == "finished"
    assert response.headers["X-Request-Id"] == body["id"]


@pytest.mark.asyncio
async def test_validation_errors_are_structured() -> None:
    async with api_client(FakeInferenceBackend()) as client:
        unsupported = await client.post("/translate", json={"text": ["hello"], "language": ["ra"]})
        too_many = await client.post(
            "/translate",
            json={"text": ["hello"] * 121, "language": ["en"]},
        )
        malformed = await client.post("/translate", json={"text": [], "language": ["en"]})

    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "validation_error"
    assert "不支持的目标语言：ra" in unsupported.json()["error"]["message"]
    assert too_many.status_code == 422
    assert malformed.status_code == 422
    assert malformed.json()["error"]["message"] == "请求参数校验失败：text: 至少需要 1 项"


@pytest.mark.asyncio
async def test_backend_status_mapping() -> None:
    async with api_client(FakeInferenceBackend(healthy=False)) as client:
        unavailable = await client.post("/translate", json={"text": ["hello"], "language": ["zh"]})
    assert unavailable.status_code == 503

    async with api_client(FakeInferenceBackend(fail_texts={"bad"})) as client:
        failed = await client.post("/translate", json={"text": ["bad"], "language": ["zh"]})
    assert failed.status_code == 502
    assert "contents" not in failed.json()

    async with api_client(
        FakeInferenceBackend(delays={"slow": 1}),
        settings_for_test(timeout=0.01),
    ) as client:
        timed_out = await client.post("/translate", json={"text": ["slow"], "language": ["zh"]})
    assert timed_out.status_code == 504


@pytest.mark.asyncio
async def test_65th_concurrent_request_is_rejected() -> None:
    backend = FakeInferenceBackend(delays={"slow": 0.1})
    async with api_client(backend, settings_for_test(max_concurrent=64)) as client:
        active = [
            asyncio.create_task(
                client.post(
                    "/translate",
                    json={"text": ["slow"], "language": ["en"]},
                )
            )
            for _ in range(64)
        ]
        await backend.started.wait()
        rejected = await client.post("/translate", json={"text": ["slow"], "language": ["en"]})
        completed = await asyncio.gather(*active)

    assert rejected.status_code == 429
    assert all(response.status_code == 200 for response in completed)


@pytest.mark.asyncio
async def test_health_metrics_and_shutdown() -> None:
    backend = FakeInferenceBackend()
    async with api_client(backend) as client:
        live = await client.get("/health/live")
        ready = await client.get("/health/ready")
        await client.post("/translate", json={"text": ["hello"], "language": ["en"]})
        metrics = await client.get("/metrics")
        openapi = await client.get("/openapi.json")

    assert live.json() == {"status": "alive"}
    assert ready.json() == {"status": "ready"}
    assert "transflow_requests_total" in metrics.text
    assert "transflow_inference_queue_wait_seconds" in metrics.text
    assert "按状态统计的翻译 API 请求数" in metrics.text
    translate_schema = openapi.json()["paths"]["/translate"]["post"]
    assert translate_schema["summary"] == "翻译文本"
    assert translate_schema["responses"]["200"]["description"] == "完整翻译结果"
    assert openapi.json()["components"]["schemas"]["TranslateRequest"]["title"] == "翻译请求"
    assert backend.closed


@pytest.mark.asyncio
async def test_readiness_recovers_when_backend_becomes_healthy() -> None:
    backend = FakeInferenceBackend(healthy=False)
    async with api_client(backend) as client:
        cold = await client.get("/health/ready")
        backend.healthy = True
        recovered = await client.get("/health/ready")

    assert cold.status_code == 503
    assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_logs_do_not_contain_text_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    private_text = "PRIVATE-SOURCE-CONTENT"
    async with api_client(FakeInferenceBackend()) as client:
        response = await client.post(
            "/translate", json={"text": [private_text], "language": ["en"]}
        )

    assert response.status_code == 200
    output = capsys.readouterr().out
    assert "translation_completed" in output
    assert private_text not in output
