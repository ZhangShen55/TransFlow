# ruff: noqa: RUF001

from collections.abc import Iterable
from types import MappingProxyType
from typing import Final

SUPPORTED_LANGUAGES: Final = MappingProxyType(
    {
        "zh": "中文",
        "en": "英语",
        "fr": "法语",
        "pt": "葡萄牙语",
        "es": "西班牙语",
        "ja": "日语",
        "tr": "土耳其语",
        "ru": "俄语",
        "ar": "阿拉伯语",
        "ko": "韩语",
        "th": "泰语",
        "it": "意大利语",
        "de": "德语",
        "vi": "越南语",
        "ms": "马来语",
        "id": "印度尼西亚语",
        "tl": "菲律宾语",
        "hi": "印地语",
        "zh-Hant": "繁体中文",
        "pl": "波兰语",
        "cs": "捷克语",
        "nl": "荷兰语",
        "km": "高棉语",
        "my": "缅甸语",
        "fa": "波斯语",
        "gu": "古吉拉特语",
        "ur": "乌尔都语",
        "te": "泰卢固语",
        "mr": "马拉地语",
        "he": "希伯来语",
        "bn": "孟加拉语",
        "ta": "泰米尔语",
        "uk": "乌克兰语",
        "bo": "藏语",
        "kk": "哈萨克语",
        "mn": "蒙古语",
        "ug": "维吾尔语",
        "yue": "粤语",
    }
)

LANGUAGE_ALIASES: Final = MappingProxyType(
    {
        # 部分上游系统将阿拉伯语错误写作 ra, 这里兼容后统一使用标准代码 ar。
        "ra": "ar",
    }
)


def normalize_language_code(code: str) -> str:
    """将上游别名转换为服务内部使用的标准语言代码。"""
    return LANGUAGE_ALIASES.get(code, code)


def target_language_name(code: str) -> str:
    normalized = normalize_language_code(code)
    try:
        return SUPPORTED_LANGUAGES[normalized]
    except KeyError as exc:
        raise ValueError(f"不支持的目标语言：{code}") from exc


def validate_language_codes(codes: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_language_code(code) for code in codes)
    duplicates = sorted({code for code in normalized if normalized.count(code) > 1})
    if duplicates:
        raise ValueError(f"目标语言重复：{', '.join(duplicates)}")
    unsupported = [code for code in normalized if code not in SUPPORTED_LANGUAGES]
    if unsupported:
        raise ValueError(f"不支持的目标语言：{', '.join(unsupported)}")
    return normalized
