# ruff: noqa: RUF001


def build_translation_prompt(source_text: str, target_language_name: str) -> str:
    return (
        f"将以下文本翻译为 {target_language_name}，"
        "注意只需要输出翻译后的结果，不要额外解释：\n\n"
        f"{source_text}"
    )
