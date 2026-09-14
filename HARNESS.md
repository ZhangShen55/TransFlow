# TransFlow Harness 记录

本文档记录代码代理和维护者复现当前项目状态所需的信息。功能契约以 OpenSpec 为准，运行参数以 `config.toml` 为准，本文档不替代二者。

## 基础环境

- 记录日期：2026-09-10
- 项目目录：`/root/workspace/TransFlow`
- Python：3.12
- Conda 环境：`transflow`
- Web 框架：FastAPI 0.116.1
- 推理服务：vLLM 0.28.0
- 模型：Hz-MT2
- 模型源目录：`/var/model_llm/Hz-MT2`
- 项目模型入口：`model/Hz-MT2`
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
| `scheduler.max_inflight_sequences` | 512 | 全局在途推理序列数 |
| `inference.batch_size` | 32 | 单个 vLLM 批请求的会话数 |
| `inference.batch_workers` | 16 | 并行批请求工作协程数 |
| `inference.max_connections` | 16 | vLLM HTTP 连接池上限 |
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

最近一次代码结果为 40 项测试通过、1 项真实模型测试按环境条件跳过，Ruff 和 Mypy 检查通过。

最大扇出验收配置为 64 个并发请求，每个请求包含 120 条非空文本和 6 个目标语言，共 46,080 个模型序列。结果为 64/64 个 HTTP 200，无超时、取消、OOM 或输出结构错误，总耗时 92.325 秒，P95 为 92.300 秒，观测峰值显存为 27,788 MiB。

2026-09-10 完成 64 并发、6,400 请求持续负载测试。每个请求包含 60 条非空混合源语言文本和 6 个目标语言，共处理 2,304,000 个翻译单元。结果为 6,400/6,400 个 HTTP 200，总耗时 4,652.029 秒，吞吐 1.376 请求/秒，P95 为 47.462 秒，P99 为 47.643 秒，无超时、取消、OOM 或输出结构错误。详细结果见 `docs/load-test-6400-c64-text60-lang6.md`。

2026-09-10 将默认在途序列调整为 256，使用 12 种源语言的 60 条真实文本完成 64 并发、640 请求测试，共处理 230,400 个翻译单元。结果为 640/640 个 HTTP 200，总耗时 533.013 秒，P95 为 54.039 秒；KV Cache 峰值 2.797%，运行序列峰值 238，等待序列峰值 0，无超时、OOM、结构错误或容器重启。详细结果见 `docs/load-test-640-c64-text60-lang6-seqs256.md`。

2026-09-10 使用相同多语种负载完成 `max_num_seqs=128/192/256/384/512/640/768/1024` A/B 测试，每档为 64 并发、128 请求。512 达到效率峰值，输出吞吐 15,712.520 token/秒，P95 为 46.320 秒，KV Cache 峰值 5.160%；640、768、1024 连续出现吞吐下降和延迟上升。全部候选均无错误、OOM 或 preemption，因此将 512、16 个批工作协程和 16 个 HTTP 连接设为默认拉起配置。详细结果见 `docs/ab-max-num-seqs-c64-r128.md`。

详细压测数据见 `docs/benchmark-results.md`，已提交的真实课程测试原始 JSON 报告位于 `docs/reports/`。

2026-09-14 使用同一节课程的真实转写样本完成 60、70、80 条文本/请求的 64 并发、600 秒持续测试。每条文本翻译为 6 种目标语言，vLLM 使用约 16GB 显存预算。三档分别完成 897、768、651 个请求，全部 HTTP 200；KV Cache 峰值为 10.6%-10.8%，无超时、取消、OOM 或 preemption。按 335 个片段折算，该重复短文本基准下 4090D 约为 900-950 节课程等效量/小时，不用于多课程低缓存容量承诺。详细结果见 `docs/course-real-load-test-16gb.md`。

同日补充多课程混合低前缀缓存测试：构造 64 个互不相同的 60 条请求，6 种目标语言共 23,040 个翻译单元。64 并发一次性处理耗时 98.390 秒，64/64 HTTP 200，前缀缓存命中率 32.097%，KV Cache 峰值 13.612%，无抢占或 OOM。按本轮工作量折算，4090D 约 419 节 335 段课程等效量/小时；据此估算独占 3090 约 189-252 节、4070 Ti Super 约 210-273 节/小时，均需目标硬件复测。详细结果见 `docs/benchmark-results.md` 和 `docs/reports/course-mixed-c64-r64.json`。

## 当前部署

- API：`http://127.0.0.1:8001`
- 接口文档：`http://127.0.0.1:8001/docs`
- 存活检查：`http://127.0.0.1:8001/health/live`
- 就绪检查：`http://127.0.0.1:8001/health/ready`
- 指标：`http://127.0.0.1:8001/metrics`

当前运行实例使用 Compose 默认端口 `8001`；也可以通过 `TRANSFLOW_API_PORT` 覆盖宿主机发布端口。

## 维护约束

- 项目文档、代码注释、公开错误信息和 OpenSpec 产物使用简体中文。
- OpenSpec 规格保留解析器要求的 `Requirement`、`Scenario`、`WHEN`、`THEN`、`MUST` 等固定关键字。
- `ar` 是阿拉伯语代码，`ra` 必须返回校验错误。
- 不提交模型权重、压测原始报告、密钥、缓存或构建产物。
- 修改调度、公平性、超时、取消、批响应重排或结果矩阵逻辑时，必须补充对应测试。
- 升级 vLLM、Transformers、PyTorch 或模型文件前，必须重新执行真实模型冒烟与最大形状测试。

## 后续记录规则

每次影响运行契约、部署拓扑、容量参数、模型版本或验收结果的变更，都应同步更新本文档。常规重构和不改变行为的格式调整无需新增记录。
