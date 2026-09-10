import os

import httpx
import pytest

MODEL_BASE_URL = os.getenv("TRANSFLOW_TEST_BASE_URL")
pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not MODEL_BASE_URL,
        reason="TRANSFLOW_TEST_BASE_URL is not set",
    ),
]


@pytest.mark.asyncio
async def test_real_model_multilingual_120_position_contract() -> None:
    assert MODEL_BASE_URL is not None
    text = [""] * 120
    text[1] = "今天天气真好。"
    text[60] = "Hello"
    text[118] = "Bonjour"
    async with httpx.AsyncClient(base_url=MODEL_BASE_URL, timeout=180) as client:
        ready = await client.get("/health/ready")
        response = await client.post(
            "/translate",
            json={"text": text, "language": ["en", "ar"]},
        )

    assert ready.status_code == 200
    assert response.status_code == 200, response.text
    contents = response.json()["result"]["contents"]
    assert [entry["language"] for entry in contents] == ["en", "ar"]
    assert all(len(entry["content"]) == 120 for entry in contents)
    for entry in contents:
        assert entry["content"][0] == ""
        assert entry["content"][119] == ""
        assert entry["content"][1]
        assert entry["content"][60]
        assert entry["content"][118]
