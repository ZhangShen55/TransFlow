# ruff: noqa: RUF002


class TransFlowError(Exception):
    """具有明确 HTTP 映射的故障基类。"""


class AdmissionRejected(TransFlowError):
    """请求准入容量已经耗尽。"""


class BackendUnavailable(TransFlowError):
    """配置的推理后端均未就绪。"""


class BackendFailure(TransFlowError):
    """推理失败，未生成可用译文。"""


class TranslationTimeout(TransFlowError):
    """请求端到端期限已到期。"""


class SchedulerClosed(TransFlowError):
    """调度器不再接受任务。"""
