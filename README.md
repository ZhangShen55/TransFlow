# TransFlow

TransFlow 是面向本地 Hy-MT2-1.8B 模型的 FastAPI 翻译编排服务。公开 API 会保持字幕式文本数组的顺序和长度，vLLM 则在 GPU 上对各个独立翻译单元进行连续批处理。

## 本地运行

```bash
conda activate transflow
python -m pip install -e '.[dev]'
uvicorn transflow.main:app --host 0.0.0.0 --port 8000
```

API 默认读取当前工作目录下的 `config.toml`。部署参数可以通过嵌套环境变量覆盖，例如：

```bash
export TRANSFLOW_INFERENCE__BASE_URLS='["http://127.0.0.1:8001/v1"]'
```

## 模型服务

将模型文件放置或链接到 `model/Hy-MT2-1.8B`，然后运行已验证的镜像：

```bash
docker run --rm --gpus device=0 \
  -v "$PWD/model/Hy-MT2-1.8B:/model:ro" \
  -p 127.0.0.1:8001:8000 \
  vllm/vllm-openai:v0.28.0 /model \
  --served-model-name Hy-MT2-1.8B \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.5 \
  --max-num-seqs 512
```

vLLM 会在冷启动期间编译并捕获 CUDA 图，整个过程可能超过一分钟。在模型端点健康前，`/health/ready` 会保持未就绪状态。

## 翻译接口

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": ["", "最后几周啊，大家。", "Hello", ""],
    "language": ["zh", "en", "fr", "es", "ru", "ar"]
  }'
```

成功响应会为每个目标语言返回一个 `content` 数组，并且数组位置与输入文本逐一对应：

```json
{
  "model": "seaCraft-translation-model",
  "id": "seaCraft-...",
  "result": {
    "contents": [
      {"content": ["", "...", "...", ""], "language": "en"},
      {"content": ["", "...", "...", ""], "language": "ar"}
    ],
    "finished_time": 0,
    "process_time_ms": 0,
    "finished_reason": "finished"
  }
}
```

上例省略了其他目标语言的结果。阿拉伯语代码为 `ar`，`ra` 会被拒绝。输入中的精确空字符串会原样返回，且不会发送到模型。

经过模式校验的完整示例位于 `docs/examples/translate-request.json` 和 `docs/examples/translate-response.json`。

## 质量检查

```bash
ruff check src tests
mypy
pytest -q
```

真实模型测试和负载测试分别使用 `model`、`load` 标记，需要显式选择。容量测试结果见 `docs/benchmark-results.md`。

## Docker Compose 部署

完整部署参数和运维命令见 `docker/README.md`。

默认部署只构建 API 镜像，vLLM 服务直接使用已验证的官方镜像：

```bash
docker compose -f docker/compose.yaml up --build
```

模型路径默认为 `model/Hy-MT2-1.8B`。也可以通过环境变量直接挂载已有模型目录，无需复制权重：

```bash
TRANSFLOW_MODEL_PATH=/var/model_llm/Hy-MT2-1.8B \
  docker compose -f docker/compose.yaml up --build
```

默认仅公开 FastAPI 端口。完成容量测试后，如需在 GPU 0 和 GPU 1 上运行两个独立模型副本，可合并多 GPU 配置：

```bash
docker compose \
  -f docker/compose.yaml \
  -f docker/compose.multi-gpu.yaml \
  up --build
```

## Harness 记录

项目的开发环境、关键实现决策、验证结果和后续维护约束记录在 `HARNESS.md`。
