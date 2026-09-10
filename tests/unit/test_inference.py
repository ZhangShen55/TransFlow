import asyncio
import json
import time

import httpx
import pytest

from transflow.core.config import GenerationSettings, InferenceSettings
from transflow.domain.errors import BackendFailure, BackendUnavailable, TranslationTimeout
from transflow.domain.work import TranslationUnit
from transflow.inference.fake import FakeInferenceBackend
from transflow.inference.vllm import VLLMInferenceBackend


class BlockingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("阻塞传输任务应当被取消")


def make_unit(
    text: str = "hello", *, request_id: str = "req", deadline: float | None = None
) -> TranslationUnit:
    return TranslationUnit(
        request_id=request_id,
        backend_request_id=f"{request_id}:0:0",
        text_index=0,
        language_index=0,
        source_text=text,
        target_language_code="zh",
        target_language_name="中文",
        prompt=f"translate {text}",
        deadline=deadline or time.monotonic() + 5,
    )


@pytest.mark.asyncio
async def test_fake_backend_supports_delay_failure_and_abort() -> None:
    backend = FakeInferenceBackend(delays={"slow": 10}, fail_texts={"bad"})

    with pytest.raises(BackendFailure):
        await backend.generate(make_unit("bad"))
    task = backend.generate(make_unit("slow", request_id="abort"))
    await asyncio.sleep(0)
    await backend.abort("abort:0:0")

    assert task.cancelled()
    assert "abort:0:0" in backend.aborted


@pytest.mark.asyncio
async def test_vllm_health_and_generation_payload() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        seen_payload.update(json.loads(request.content))
        assert request.headers["X-Request-Id"].startswith("req:batch:")
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": "你好"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMInferenceBackend(
        InferenceSettings(base_urls=("http://vllm:8000/v1",)),
        GenerationSettings(),
        client=client,
    )

    assert await backend.is_healthy()
    result = await backend.generate(make_unit())
    assert result.text == "你好"
    assert result.prompt_tokens == 4
    assert seen_payload["temperature"] == 0.0
    assert seen_payload["top_k"] == -1
    assert seen_payload["messages"] == [[{"role": "user", "content": "translate hello"}]]
    await backend.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_retries_connection_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("not connected", request=request)
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "ok"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMInferenceBackend(
        InferenceSettings(
            base_urls=("http://vllm-1:8000/v1", "http://vllm-2:8000/v1"),
            max_retries=1,
        ),
        GenerationSettings(),
        client=client,
    )

    assert (await backend.generate(make_unit())).text == "ok"
    assert calls == 2
    await backend.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_does_not_retry_read_timeout() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("generation timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMInferenceBackend(
        InferenceSettings(max_retries=3), GenerationSettings(), client=client
    )

    with pytest.raises(TranslationTimeout):
        await backend.generate(make_unit())
    assert calls == 1
    await backend.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_rejects_malformed_and_unavailable_responses() -> None:
    responses = iter(
        [
            httpx.Response(200, json={"choices": []}),
            httpx.Response(503),
            httpx.Response(503),
        ]
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: next(responses)))
    backend = VLLMInferenceBackend(
        InferenceSettings(max_retries=1), GenerationSettings(), client=client
    )

    with pytest.raises(BackendFailure):
        await backend.generate(make_unit(request_id="malformed"))
    with pytest.raises(BackendUnavailable):
        await backend.generate(make_unit(request_id="unavailable"))
    await backend.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_unready_health_and_active_cancellation() -> None:
    unhealthy_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    )
    unhealthy = VLLMInferenceBackend(
        InferenceSettings(), GenerationSettings(), client=unhealthy_client
    )
    assert not await unhealthy.is_healthy()
    await unhealthy.close()
    await unhealthy_client.aclose()

    transport = BlockingTransport()
    blocking_client = httpx.AsyncClient(transport=transport)
    backend = VLLMInferenceBackend(
        InferenceSettings(), GenerationSettings(), client=blocking_client
    )
    task = backend.generate(make_unit(request_id="cancel-vllm"))
    await transport.started.wait()

    await backend.abort("cancel-vllm:0:0")

    assert task.cancelled()
    await backend.close()
    await blocking_client.aclose()


@pytest.mark.asyncio
async def test_vllm_micro_batches_and_reorders_choices() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        assert len(payload["messages"]) == 2
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 1, "message": {"content": "second"}},
                    {"index": 0, "message": {"content": "first"}},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = VLLMInferenceBackend(
        InferenceSettings(
            batch_size=2,
            batch_workers=1,
            batch_wait_milliseconds=10,
            max_connections=1,
        ),
        GenerationSettings(),
        client=client,
    )

    first, second = await asyncio.gather(
        backend.generate(make_unit("one", request_id="one")),
        backend.generate(make_unit("two", request_id="two")),
    )

    assert [first.text, second.text] == ["first", "second"]
    assert calls == 1
    await backend.close()
    await client.aclose()
