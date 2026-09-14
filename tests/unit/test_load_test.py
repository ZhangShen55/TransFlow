from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "load_test.py"
SPEC = importlib.util.spec_from_file_location("transflow_load_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LOAD_TEST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOAD_TEST
SPEC.loader.exec_module(LOAD_TEST)


def test_oom_detection_does_not_scan_successful_translation_text() -> None:
    assert not LOAD_TEST.response_reports_oom(200, "There is room for tomorrow.")
    assert not LOAD_TEST.response_reports_oom(200, "OOM")
    assert LOAD_TEST.response_reports_oom(500, "CUDA out of memory")
    assert LOAD_TEST.response_reports_oom(503, "worker OOM")


def test_load_texts_accepts_json_fragment(tmp_path: Path) -> None:
    source = tmp_path / "texts.json"
    source.write_text('"text": ["一", "two", "ثلاثة"]', encoding="utf-8")

    assert LOAD_TEST.load_texts(source, 2) == ["一", "two"]


def test_load_texts_accepts_course_segments(tmp_path: Path) -> None:
    source = tmp_path / "course.json"
    source.write_text(
        '{"segments": [{"text": "第一句"}, {"text": "second"}]}',
        encoding="utf-8",
    )

    assert LOAD_TEST.load_texts(source, None) == ["第一句", "second"]
