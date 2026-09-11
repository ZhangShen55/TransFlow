# Docker 部署说明

本目录提供 TransFlow 的 API 镜像和 Docker Compose 部署文件。FastAPI 与 vLLM 分别运行在独立容器中，模型权重只读挂载到 vLLM，`config.toml` 只读挂载到 API，默认仅向宿主机公开 FastAPI 端口。

Compose 使用固定名称的 `transflow_edge` 和 `transflow_inference` 网络。网络已存在时直接复用，不存在时自动创建，因此不需要手工创建网络，也不要将它们设置为 `external`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `Dockerfile.api` | 构建轻量 FastAPI 镜像，不包含 CUDA、vLLM 或模型权重 |
| `compose.yaml` | 单 GPU 默认部署，一个 API 容器和一个 vLLM 容器 |
| `compose.multi-gpu.yaml` | 双 GPU 覆盖配置，增加第二个独立 vLLM 副本 |

## 环境要求

- Docker Engine 支持 Compose V2，即可以执行 `docker compose version`。
- 已安装 NVIDIA 驱动和 NVIDIA Container Toolkit。
- `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` 能识别目标 GPU。
- 模型目录包含完整的 Hz-MT2 权重、分词器和配置文件。
- 所有命令均从项目根目录执行。

## 模型目录

默认模型路径为项目内的 `model/Hz-MT2`。模型内容已被 Git 和 Docker 构建上下文忽略，不会提交到仓库或复制进 API 镜像。

如果模型位于 `/var/model_llm/Hz-MT2`，启动时通过环境变量直接挂载：

```bash
export TRANSFLOW_MODEL_PATH=/var/model_llm/Hz-MT2
```

Compose 会将该目录以只读方式挂载到 vLLM 容器的 `/model`。

API 容器会将项目根目录的 `config.toml` 以只读方式挂载到 `/app/config.toml`。镜像构建时也会复制一份配置作为未使用 Compose 挂载时的后备；通过 Compose 运行时以宿主机文件为准。修改配置后需要重新创建 API 容器：

```bash
TRANSFLOW_API_PORT=18000 \
docker compose -f docker/compose.yaml up -d --no-deps --force-recreate api
```

## 单 GPU 启动

默认使用 GPU 0，并将 FastAPI 发布到宿主机的 8000 端口：

```bash
TRANSFLOW_MODEL_PATH=/var/model_llm/Hz-MT2 \
docker compose -f docker/compose.yaml up -d --build --wait
```

如果需要使用其他端口，可以通过 `TRANSFLOW_API_PORT` 覆盖；例如使用 18000 端口：

```bash
TRANSFLOW_MODEL_PATH=/var/model_llm/Hz-MT2 \
TRANSFLOW_GPU_ID=0 \
TRANSFLOW_API_PORT=18000 \
docker compose -f docker/compose.yaml up -d --build --wait
```

vLLM 冷启动需要加载权重、编译并捕获 CUDA 图，通常超过一分钟。`--wait` 会一直等待 vLLM 和 API 健康检查通过。

## 默认容量

默认值来自本机 RTX 4090 上的 64 并发多语种 A/B 测试：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `VLLM_MAX_NUM_SEQS` | 512 | vLLM 同时调度的最大序列数 |
| `TRANSFLOW_MAX_INFLIGHT_SEQUENCES` | 512 | API 全局进入推理阶段的序列数 |
| `TRANSFLOW_BATCH_WORKERS` | 16 | 并行提交 vLLM 微批的工作协程数 |
| `TRANSFLOW_MAX_CONNECTIONS` | 16 | API 到 vLLM 的 HTTP 连接池上限 |
| `TRANSFLOW_DISPATCH_CHUNK_SIZE` | 60 | 每个请求一轮释放的文本位置数 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.50 | vLLM 可使用的 GPU 显存比例 |

前四个参数必须协同调整。每个微批最多包含 32 个序列，因此必须满足：

```text
TRANSFLOW_BATCH_WORKERS * 32 >= TRANSFLOW_MAX_INFLIGHT_SEQUENCES
TRANSFLOW_MAX_CONNECTIONS >= TRANSFLOW_BATCH_WORKERS
TRANSFLOW_MAX_INFLIGHT_SEQUENCES <= VLLM_MAX_NUM_SEQS
```

例如临时回退到 256：

```bash
VLLM_MAX_NUM_SEQS=256 \
TRANSFLOW_MAX_INFLIGHT_SEQUENCES=256 \
TRANSFLOW_BATCH_WORKERS=8 \
TRANSFLOW_MAX_CONNECTIONS=8 \
TRANSFLOW_MODEL_PATH=/var/model_llm/Hz-MT2 \
docker compose -f docker/compose.yaml up -d --force-recreate --wait
```

不要只提高 `VLLM_MAX_NUM_SEQS`。如果 API 在途上限、工作协程或连接池保持较小值，新增的 vLLM 容量不会被使用。

## 可用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TRANSFLOW_MODEL_PATH` | `../model/Hz-MT2` | 宿主机模型目录 |
| `TRANSFLOW_GPU_ID` | `0` | 单 GPU 部署使用的设备编号 |
| `TRANSFLOW_API_BIND` | `0.0.0.0` | API 在宿主机上的绑定地址 |
| `TRANSFLOW_API_PORT` | `8000` | API 在宿主机上的发布端口 |
| `VLLM_API_KEY` | `local` | API 容器调用 vLLM 的内部密钥 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.50` | vLLM GPU 显存利用率 |
| `VLLM_MAX_NUM_SEQS` | `512` | vLLM 最大运行序列数 |
| `TRANSFLOW_MAX_INFLIGHT_SEQUENCES` | `512` | API 全局在途序列上限 |
| `TRANSFLOW_BATCH_WORKERS` | `16` | vLLM 微批工作协程数 |
| `TRANSFLOW_MAX_CONNECTIONS` | `16` | vLLM HTTP 连接池上限 |
| `TRANSFLOW_DISPATCH_CHUNK_SIZE` | `60` | 文本位置调度分片大小 |
| `TRANSFLOW_LOG_LEVEL` | `INFO` | API 日志级别 |

其他业务配置继续由项目根目录的 `config.toml` 管理。Compose 环境变量会覆盖其中对应字段。

## 服务验证

查看容器状态：

```bash
docker compose -f docker/compose.yaml ps
```

检查 API：

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/metrics
```

使用 18000 端口启动时，将上述地址中的 8000 替换为 18000。接口文档位于 `/docs`。

发送翻译请求：

```bash
curl -fsS http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": ["", "最后几周啊，大家。", "Hello"],
    "language": ["zh", "en", "fr", "es", "ru", "ar"]
  }'
```

vLLM 位于 Compose 私有网络，不向宿主机发布端口。可以在容器内检查其健康状态：

```bash
docker exec transflow-vllm-1 \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

## 日志与重启

查看全部日志：

```bash
docker compose -f docker/compose.yaml logs -f
```

只查看一个服务：

```bash
docker compose -f docker/compose.yaml logs -f api
docker compose -f docker/compose.yaml logs -f vllm
```

配置或模型挂载发生变化后，重新创建容器：

```bash
docker compose -f docker/compose.yaml up -d --build --force-recreate --wait
```

停止并删除容器和网络，不会删除宿主机模型：

```bash
docker compose -f docker/compose.yaml down
```

## 双 GPU 部署

双 GPU 覆盖会在 GPU 0 和 GPU 1 上分别运行一个模型副本，API 按轮询方式访问两个 vLLM 地址：

```bash
TRANSFLOW_MODEL_PATH=/var/model_llm/Hz-MT2 \
TRANSFLOW_GPU_0=0 \
TRANSFLOW_GPU_1=1 \
docker compose \
  -f docker/compose.yaml \
  -f docker/compose.multi-gpu.yaml \
  up -d --build --wait
```

每个副本默认使用 `max_num_seqs=512`。API 的 512 在途序列是两个副本共享的应用上限；提高总在途容量前必须重新进行双 GPU 压测。

停止双 GPU 部署时必须使用相同的配置文件组合：

```bash
docker compose \
  -f docker/compose.yaml \
  -f docker/compose.multi-gpu.yaml \
  down
```

## 常见问题

### API 长时间处于未就绪状态

先执行 `docker compose -f docker/compose.yaml ps` 和 `docker compose -f docker/compose.yaml logs vllm`。首次启动通常需要一分钟以上；如果 vLLM 发生 OOM，应降低 `VLLM_GPU_MEMORY_UTILIZATION`、`VLLM_MAX_NUM_SEQS` 或排查同一 GPU 上的其他进程。

### 请求返回 429

API 已达到 `max_concurrent_requests=64` 的准入上限。该限制保护模型服务，不应通过增加 Uvicorn worker 绕过；需要调整容量时必须重新压测。

### Compose 提示配置校验失败

检查批处理容量和连接数是否满足前述关系。也可以在启动前执行：

```bash
docker compose -f docker/compose.yaml config --quiet
docker compose \
  -f docker/compose.yaml \
  -f docker/compose.multi-gpu.yaml \
  config --quiet
```

### 修改 vLLM 镜像版本

当前镜像 `vllm/vllm-openai:v0.28.0` 已针对 Hz-MT2 验证。升级镜像、PyTorch、Transformers 或模型文件后，必须重新执行真实模型测试和容量测试。
