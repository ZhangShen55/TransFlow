from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from transflow.domain.work import GenerationResult
from transflow.scheduler.fair import SchedulerSnapshot


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "transflow_requests_total",
            "按状态统计的翻译 API 请求数",
            ("status",),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "transflow_request_duration_seconds",
            "翻译请求端到端耗时",
            registry=self.registry,
        )
        self.request_text_items = Histogram(
            "transflow_request_text_items",
            "每个翻译请求的文本条目数",
            buckets=(1, 8, 16, 32, 60, 120, 240),
            registry=self.registry,
        )
        self.request_languages = Histogram(
            "transflow_request_languages",
            "每个翻译请求的目标语言数",
            buckets=(1, 2, 3, 4, 6, 8, 16),
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "transflow_inference_queue_wait_seconds",
            "翻译单元等待推理容量的时间",
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "transflow_inference_duration_seconds",
            "按结果统计的推理耗时",
            ("outcome",),
            registry=self.registry,
        )
        self.inference_units = Counter(
            "transflow_inference_units_total",
            "按结果统计的推理单元数",
            ("outcome",),
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "transflow_prompt_tokens_total",
            "后端报告的输入 token 数",
            registry=self.registry,
        )
        self.completion_tokens = Counter(
            "transflow_completion_tokens_total",
            "后端报告的输出 token 数",
            registry=self.registry,
        )
        self.pending_units = Gauge(
            "transflow_scheduler_pending_units",
            "待处理推理单元数",
            registry=self.registry,
        )
        self.active_units = Gauge(
            "transflow_scheduler_active_units",
            "活动推理单元数",
            registry=self.registry,
        )
        self.pending_requests = Gauge(
            "transflow_scheduler_pending_requests",
            "存在待处理推理单元的请求数",
            registry=self.registry,
        )

    def record_request(
        self,
        *,
        status: int,
        seconds: float,
        text_items: int,
        languages: int,
    ) -> None:
        self.requests.labels(status=str(status)).inc()
        self.request_duration.observe(seconds)
        self.request_text_items.observe(text_items)
        self.request_languages.observe(languages)

    def update_scheduler(self, snapshot: SchedulerSnapshot) -> None:
        self.pending_units.set(snapshot.pending)
        self.active_units.set(snapshot.active)
        self.pending_requests.set(snapshot.requests)

    def record_queue_wait(self, seconds: float) -> None:
        self.queue_wait.observe(seconds)

    def record_inference(
        self,
        seconds: float,
        outcome: str,
        result: GenerationResult | None,
    ) -> None:
        self.inference_duration.labels(outcome=outcome).observe(seconds)
        self.inference_units.labels(outcome=outcome).inc()
        if result is not None:
            self.prompt_tokens.inc(result.prompt_tokens)
            self.completion_tokens.inc(result.completion_tokens)

    def render(self) -> bytes:
        return generate_latest(self.registry)
