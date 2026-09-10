import json
from pathlib import Path

from transflow.schemas.translation import TranslateRequest, TranslateResponse

EXAMPLES = Path(__file__).parents[2] / "docs" / "examples"


def test_documented_request_and_response_match_schemas() -> None:
    request = TranslateRequest.model_validate_json(
        (EXAMPLES / "translate-request.json").read_text()
    )
    response = TranslateResponse.model_validate_json(
        (EXAMPLES / "translate-response.json").read_text()
    )

    assert [content.language for content in response.result.contents] == request.language
    assert all(len(content.content) == len(request.text) for content in response.result.contents)
    assert json.loads((EXAMPLES / "translate-request.json").read_text())["language"][-1] == "ar"
