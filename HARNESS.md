# TransFlow Harness 记录

本文档记录代码代理和维护者复现当前项目状态所需的信息。功能契约以 OpenSpec 为准，运行参数以 `config.toml` 为准，本文档不替代二者。

## 基础环境

- 记录日期：2026-09-10
- 项目目录：`/root/workspace/TransFlow`
- Python：3.12
- Conda 环境：`transflow`
- Web 框架：FastAPI 0.116.1
- 推理服务：vLLM 0.28.0
- 模型：Hy-MT2-1.8B
- 模型源目录：`/var/model_llm/Hy-MT2-1.8B`
- 项目模型入口：`model/Hy-MT2-1.8B`
- vLLM 镜像：`vllm/vllm-openai:v0.28.0`
- 已验证镜像摘要前缀：`sha256:609a5b463503`

## 架构记录

- FastAPI 和 vLLM 使用独立容器，API 镜像不包含 CUDA 或模型权重。
- API 通过 Compose 私有网络调用 `http://vllm:8000/v1`，默认不向宿主机发布 vLLM 端口。
- 每个非空的“文本索引 + 目标语言”组合是独立翻译单元，由 vLLM 连续批处理。
- 应用使用“目标语言索引 x 文本索引”矩阵重组结果，不依赖模型生成分隔符。
- 精确空字符串在本地完成，不调用模型；只包含空白字符的文本仍提交翻译。
- 调度器按请求轮询，限制准入请求、待处理单元和在途序列，并传播超时与取消。
- vLLM 客户端使用 `/v1/chat/completions/batch` 微批接口，按 `choice.index` 恢复顺序。

## 当前容量配置

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `api.max_concurrent_requests` | 64 | 同时准入的 API 请求数 |
| `api.max_text_items` | 120 | 单请求最大文本条数 |
| `api.max_target_languages` | 6 | 单请求最大目标语言数 |
| `scheduler.dispatch_chunk_size` | 60 | 每轮释放的文本位置数 |
| `scheduler.max_inflight_sequences` | 128 | 全局在途推理序列数 |
| `inference.batch_size` | 32 | 单个 vLLM 批请求的会话数 |
| `inference.batch_workers` | 4 | 并行批请求工作协程数 |
| `api.request_timeout_seconds` | 600 | API 端到端期限（秒） |

这些值来自本机 GPU 的渐进压测，不应在服务启动时自动探测到 OOM。更换 GPU、模型版本、最大文本长度或目标语言数量后，必须重新运行容量测试。

## 验证记录

已通过以下检查：

```bash
conda run -n transflow ruff check src tests scripts
conda run -n transflow ruff format --check src tests scripts
conda run -n transflow mypy
conda run -n transflow pytest -q
TRANSFLOW_TEST_BASE_URL=http://127.0.0.1:18000 \
  conda run -n transflow pytest -q tests/model/test_real_model.py
npx --yes @fission-ai/openspec@1.13.0 \
  validate build-translation-service --strict
docker compose -f docker/compose.yaml config --quiet
docker compose -f docker/compose.yaml \
  -f docker/compose.multi-gpu.yaml config --quiet
```

最近一次完整结果为 39 项测试通过，Ruff、Mypy、Docker Compose 配置和严格 OpenSpec 校验通过。

最大扇出验收配置为 64 个并发请求，每个请求包含 120 条非空文本和 6 个目标语言，共 46,080 个模型序列。结果为 64/64 个 HTTP 200，无超时、取消、OOM 或输出结构错误，总耗时 92.325 秒，P95 为 92.300 秒，观测峰值显存为 27,788 MiB。

详细压测数据见 `docs/benchmark-results.md`，本机原始 JSON 报告位于已被 Git 忽略的 `reports/`。

## 当前部署

- API：`http://127.0.0.1:18000`
- 接口文档：`http://127.0.0.1:18000/docs`
- 存活检查：`http://127.0.0.1:18000/health/live`
- 就绪检查：`http://127.0.0.1:18000/health/ready`
- 指标：`http://127.0.0.1:18000/metrics`

端口 `18000` 是当前运行实例的部署覆盖值；代码默认端口仍为 `8000`。

## 维护约束

- 项目文档、代码注释、公开错误信息和 OpenSpec 产物使用简体中文。
- OpenSpec 规格保留解析器要求的 `Requirement`、`Scenario`、`WHEN`、`THEN`、`MUST` 等固定关键字。
- `ar` 是阿拉伯语代码，`ra` 必须返回校验错误。
- 不提交模型权重、压测原始报告、密钥、缓存或构建产物。
- 修改调度、公平性、超时、取消、批响应重排或结果矩阵逻辑时，必须补充对应测试。
- 升级 vLLM、Transformers、PyTorch 或模型文件前，必须重新执行真实模型冒烟与最大形状测试。

## 后续记录规则

每次影响运行契约、部署拓扑、容量参数、模型版本或验收结果的变更，都应同步更新本文档。常规重构和不改变行为的格式调整无需新增记录。
