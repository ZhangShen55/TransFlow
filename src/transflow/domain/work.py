from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    request_id: str
    backend_request_id: str
    text_index: int
    language_index: int
    source_text: str
    target_language_code: str
    target_language_name: str
    prompt: str
    deadline: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
