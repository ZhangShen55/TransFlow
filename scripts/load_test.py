#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF003
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

SAMPLE_TEXTS = (
    "最后几周啊，大家。",
    "This is a short subtitle segment.",
    "Ces dernières semaines, tout le monde.",
    "Esta es una frase breve.",
    "Это короткая строка субтитров.",
    "هذه جملة ترجمة قصيرة.",
)


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:", 1).replace("options:", "选项:", 1)


@dataclass
class RequestObservation:
    latency_ms: float
    status: int
    shape_violation: bool = False
    timeout: bool = False
    cancelled: bool = False
    oom: bool = False
    error: str | None = None


@dataclass
class VLLMMetricsSample:
    kv_cache_usage: float
    running_requests: float
    waiting_requests: float
    preemptions: float
    prefix_cache_queries: float
    prefix_cache_hits: float


@dataclass
class GPUSample:
    memory_mib: float
    utilization_percent: float
    power_watts: float


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[position], 3)


def metric_value(metrics: str, name: str) -> float:
    prefix = f"{name} "
    for line in metrics.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    return 0.0


def labeled_metric_value(metrics: str, name: str) -> float:
    """汇总同一指标的标签序列, 兼容单引擎与多引擎指标。"""
    prefixes = (f"{name} ", f"{name}{{")
    total = 0.0
    for line in metrics.splitlines():
        if not line.startswith(prefixes):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return total


def response_reports_oom(status: int, response_text: str) -> bool:
    """只从服务端错误中识别明确的显存不足信息。"""
    lowered = response_text.lower()
    return status >= 500 and (
        "out of memory" in lowered or re.search(r"\boom\b", lowered) is not None
    )


def build_payload(text_items: int, nonempty_items: int, languages: list[str]) -> dict[str, Any]:
    texts = [""] * text_items
    if nonempty_items:
        step = max(1, text_items // nonempty_items)
        for source_index in range(nonempty_items):
            text_index = min(source_index * step, text_items - 1)
            texts[text_index] = SAMPLE_TEXTS[source_index % len(SAMPLE_TEXTS)]
    return {"text": texts, "language": languages}


def load_texts(path: Path, limit: int | None) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        # 支持用户直接提供的 `"text": [...]` JSON 对象片段。
        parsed = json.loads("{" + raw + "}")
    if isinstance(parsed, dict):
        texts = parsed.get("text")
        if texts is None and isinstance(parsed.get("segments"), list):
            texts = [segment.get("text") for segment in parsed["segments"]]
    else:
        texts = parsed
    if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError("文本文件必须是字符串数组，或包含字符串数组 text 的 JSON 对象")
    selected = texts[:limit] if limit is not None else texts
    if not selected:
        raise ValueError("文本文件中没有可用于压测的文本")
    return selected


def valid_shape(body: Any, payload: dict[str, Any]) -> bool:
    try:
        contents = body["result"]["contents"]
        expected_languages = payload["language"]
        expected_length = len(payload["text"])
        return [entry["language"] for entry in contents] == expected_languages and all(
            len(entry["content"]) == expected_length for entry in contents
        )
    except (KeyError, TypeError):
        return False


async def gpu_sample(gpu_index: int) -> GPUSample:
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        return GPUSample(0.0, 0.0, 0.0)
    values = stdout.decode().strip().splitlines()[0].split(",")
    return GPUSample(*(float(value.strip()) for value in values))


async def monitor_gpu(gpu_index: int, stop: asyncio.Event, samples: list[GPUSample]) -> None:
    while not stop.is_set():
        samples.append(await gpu_sample(gpu_index))
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.2)
        except TimeoutError:
            continue


async def monitor_vllm(
    metrics_url: str,
    stop: asyncio.Event,
    samples: list[VLLMMetricsSample],
) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        while not stop.is_set():
            try:
                metrics = (await client.get(metrics_url)).text
                samples.append(
                    VLLMMetricsSample(
                        kv_cache_usage=labeled_metric_value(metrics, "vllm:kv_cache_usage_perc"),
                        running_requests=labeled_metric_value(metrics, "vllm:num_requests_running"),
                        waiting_requests=labeled_metric_value(metrics, "vllm:num_requests_waiting"),
                        preemptions=labeled_metric_value(metrics, "vllm:num_preemptions_total"),
                        prefix_cache_queries=labeled_metric_value(
                            metrics, "vllm:prefix_cache_queries_total"
                        ),
                        prefix_cache_hits=labeled_metric_value(
                            metrics, "vllm:prefix_cache_hits_total"
                        ),
                    )
                )
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except TimeoutError:
                continue


async def get_metrics(client: httpx.AsyncClient) -> str:
    try:
        response = await client.get("/metrics", timeout=5)
        return response.text if response.is_success else ""
    except httpx.HTTPError:
        return ""


async def send_request(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    start: asyncio.Event,
    semaphore: asyncio.Semaphore,
) -> RequestObservation:
    await start.wait()
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post("/translate", json=payload)
            latency = (time.perf_counter() - started) * 1000
            body: Any = response.json()
            response_text = response.text.lower()
            return RequestObservation(
                latency_ms=latency,
                status=response.status_code,
                shape_violation=response.status_code == 200 and not valid_shape(body, payload),
                timeout=response.status_code == 504,
                oom=response_reports_oom(response.status_code, response_text),
                error=None if response.status_code == 200 else response.text[:500],
            )
        except httpx.TimeoutException as exc:
            return RequestObservation(
                latency_ms=(time.perf_counter() - started) * 1000,
                status=0,
                timeout=True,
                error=str(exc),
            )
        except asyncio.CancelledError:
            return RequestObservation(
                latency_ms=(time.perf_counter() - started) * 1000,
                status=0,
                cancelled=True,
                error="客户端任务已取消",
            )
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return RequestObservation(
                latency_ms=(time.perf_counter() - started) * 1000,
                status=0,
                error=str(exc),
            )


async def run_profile(
    *,
    base_url: str,
    concurrency: int,
    request_count: int | None,
    duration_seconds: float | None,
    payload: dict[str, Any],
    dispatch_size: int,
    timeout_seconds: float,
    gpu_index: int | None,
    vllm_metrics_url: str | None,
) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=max(concurrency, 1),
        max_keepalive_connections=max(concurrency, 1),
    )
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout_seconds,
        limits=limits,
    ) as client:
        before_metrics = await get_metrics(client)
        start = asyncio.Event()
        semaphore = asyncio.Semaphore(concurrency)
        observations: list[RequestObservation] = []

        async def run_sustained_worker(stop: asyncio.Event) -> None:
            while not stop.is_set():
                observations.append(await send_request(client, payload, start, semaphore))

        stop_requests = asyncio.Event()
        tasks: list[asyncio.Task[Any]]
        if duration_seconds is None:
            tasks = [
                asyncio.create_task(send_request(client, payload, start, semaphore))
                for _ in range(request_count or 0)
            ]
        else:
            tasks = [
                asyncio.create_task(run_sustained_worker(stop_requests))
                for _ in range(concurrency)
            ]
        gpu_samples: list[GPUSample] = []
        stop_gpu = asyncio.Event()
        monitor = (
            asyncio.create_task(monitor_gpu(gpu_index, stop_gpu, gpu_samples))
            if gpu_index is not None
            else None
        )
        vllm_samples: list[VLLMMetricsSample] = []
        vllm_monitor = (
            asyncio.create_task(monitor_vllm(vllm_metrics_url, stop_gpu, vllm_samples))
            if vllm_metrics_url is not None
            else None
        )
        started = time.perf_counter()
        start.set()
        if duration_seconds is None:
            observations = list(await asyncio.gather(*tasks))
        else:
            await asyncio.sleep(duration_seconds)
            stop_requests.set()
            await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started
        stop_gpu.set()
        if monitor is not None:
            await monitor
        if vllm_monitor is not None:
            await vllm_monitor
        after_metrics = await get_metrics(client)
        healthy_after = (await client.get("/health/live", timeout=5)).is_success

    latencies = [observation.latency_ms for observation in observations]
    statuses = Counter(str(observation.status) for observation in observations)
    completed_request_count = len(observations)
    # Prometheus 计数器是进程累计值，报告只记录本轮压测前后的差值。
    queue_sum = metric_value(
        after_metrics, "transflow_inference_queue_wait_seconds_sum"
    ) - metric_value(before_metrics, "transflow_inference_queue_wait_seconds_sum")
    queue_count = metric_value(
        after_metrics, "transflow_inference_queue_wait_seconds_count"
    ) - metric_value(before_metrics, "transflow_inference_queue_wait_seconds_count")
    return {
        "dispatch_size": dispatch_size,
        "concurrency": concurrency,
        "request_count": completed_request_count,
        "request_target": request_count,
        "duration_seconds": duration_seconds,
        "text_items": len(payload["text"]),
        "nonempty_items": sum(value != "" for value in payload["text"]),
        "languages": payload["language"],
        "elapsed_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(completed_request_count / elapsed, 3),
        "latency_ms": {
            "min": round(min(latencies), 3),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 3),
        },
        "queue_wait_seconds": {
            "observations": int(queue_count),
            "average": round(queue_sum / queue_count, 6) if queue_count else 0.0,
        },
        "backend_tokens": {
            "prompt": int(
                metric_value(after_metrics, "transflow_prompt_tokens_total")
                - metric_value(before_metrics, "transflow_prompt_tokens_total")
            ),
            "completion": int(
                metric_value(after_metrics, "transflow_completion_tokens_total")
                - metric_value(before_metrics, "transflow_completion_tokens_total")
            ),
        },
        "peak_gpu_memory_mib": max((sample.memory_mib for sample in gpu_samples), default=0.0),
        "gpu": {
            "samples": len(gpu_samples),
            "average_utilization_percent": round(
                sum(sample.utilization_percent for sample in gpu_samples) / len(gpu_samples),
                3,
            )
            if gpu_samples
            else 0.0,
            "peak_utilization_percent": max(
                (sample.utilization_percent for sample in gpu_samples), default=0.0
            ),
            "average_power_watts": round(
                sum(sample.power_watts for sample in gpu_samples) / len(gpu_samples), 3
            )
            if gpu_samples
            else 0.0,
            "peak_power_watts": max((sample.power_watts for sample in gpu_samples), default=0.0),
        },
        "vllm": {
            "samples": len(vllm_samples),
            "peak_kv_cache_usage_percent": round(
                max((sample.kv_cache_usage for sample in vllm_samples), default=0.0) * 100,
                3,
            ),
            "peak_running_requests": int(
                max((sample.running_requests for sample in vllm_samples), default=0.0)
            ),
            "peak_waiting_requests": int(
                max((sample.waiting_requests for sample in vllm_samples), default=0.0)
            ),
            "preemptions": int(max(0.0, vllm_samples[-1].preemptions - vllm_samples[0].preemptions))
            if vllm_samples
            else 0,
            "prefix_cache_hit_rate_percent": round(
                100
                * max(
                    0.0,
                    vllm_samples[-1].prefix_cache_hits - vllm_samples[0].prefix_cache_hits,
                )
                / (vllm_samples[-1].prefix_cache_queries - vllm_samples[0].prefix_cache_queries),
                3,
            )
            if len(vllm_samples) > 1
            and vllm_samples[-1].prefix_cache_queries > vllm_samples[0].prefix_cache_queries
            else 0.0,
        },
        "status_counts": dict(sorted(statuses.items())),
        "errors": sum(observation.error is not None for observation in observations),
        "timeouts": sum(observation.timeout for observation in observations),
        "cancellations": sum(observation.cancelled for observation in observations),
        "cardinality_violations": sum(observation.shape_violation for observation in observations),
        "oom_events": sum(observation.oom for observation in observations),
        "service_healthy_after": healthy_after,
    }


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(
        description="运行 TransFlow HTTP 负载测试",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API 基础地址")
    parser.add_argument("--concurrency", default="1,8,16,32,64", help="逗号分隔的并发级别")
    parser.add_argument(
        "--requests-per-level", type=int, default=0, help="每个并发级别的请求数；0 表示与并发量相同"
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="持续压测时长，设置后每个并发 worker 会循环发送请求",
    )
    parser.add_argument(
        "--dispatch-size", type=int, required=True, help="服务使用的调度分片大小，仅用于记录"
    )
    parser.add_argument("--text-items", type=int, default=120, help="每个请求的文本位置总数")
    parser.add_argument("--nonempty-items", type=int, default=4, help="每个请求的非空文本数")
    parser.add_argument("--languages", default="zh,en,fr,es,ru,ar", help="逗号分隔的目标语言代码")
    parser.add_argument(
        "--timeout-seconds", type=float, default=900, help="单个 HTTP 请求超时时间，单位为秒"
    )
    parser.add_argument("--gpu-index", type=int, help="需要采集显存的 GPU 索引")
    parser.add_argument(
        "--text-file",
        type=Path,
        help="JSON 文本数组、包含 text 数组的文件，或包含 segments[].text 的课程转写文件",
    )
    parser.add_argument("--text-limit", type=int, help="从文本文件开头选取的最大条数")
    parser.add_argument("--vllm-metrics-url", help="用于采集 KV Cache 和请求峰值的指标地址")
    parser.add_argument("--output", type=Path, help="JSON 报告输出路径")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    if args.text_file:
        texts = load_texts(args.text_file, args.text_limit)
        payload = {"text": texts, "language": args.languages.split(",")}
    else:
        payload = build_payload(
            args.text_items,
            args.nonempty_items,
            args.languages.split(","),
        )
    profiles = []
    for concurrency in parse_csv_ints(args.concurrency):
        if args.duration_seconds is not None and args.duration_seconds <= 0:
            raise ValueError("--duration-seconds 必须大于 0")
        request_count = (
            None
            if args.duration_seconds is not None
            else (args.requests_per_level or concurrency)
        )
        profiles.append(
            await run_profile(
                base_url=args.base_url,
                concurrency=concurrency,
                request_count=request_count,
                duration_seconds=args.duration_seconds,
                payload=payload,
                dispatch_size=args.dispatch_size,
                timeout_seconds=args.timeout_seconds,
                gpu_index=args.gpu_index,
                vllm_metrics_url=args.vllm_metrics_url,
            )
        )
    return {
        "generated_at_unix": int(time.time()),
        "base_url": args.base_url,
        "profiles": profiles,
    }


def main() -> None:
    args = parse_args()
    report = asyncio.run(async_main(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
