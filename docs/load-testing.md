# 负载测试

容量测试必须在指定且无其他负载的 GPU 上运行。使用待测的调度分片大小启动服务，并在报告中记录相同数值：

```bash
TRANSFLOW_MODEL_PATH=/var/model_llm/Hz-MT2 \
TRANSFLOW_DISPATCH_CHUNK_SIZE=8 \
TRANSFLOW_API_PORT=18000 \
docker compose -f docker/compose.yaml up -d --wait --force-recreate api

conda run -n transflow python scripts/load_test.py \
  --base-url http://127.0.0.1:18000 \
  --dispatch-size 8 \
  --concurrency 1,8,16,32,64 \
  --text-items 120 \
  --nonempty-items 4 \
  --languages zh,en,fr,es,ru,ar \
  --gpu-index 0 \
  --output reports/load-dispatch-8.json
```

分别对分片大小 `16`、`32` 和 `60` 重复测试。最大请求结构包含 120 个位置和 6 个目标语言，`nonempty-items` 控制实际 GPU 工作量。在确定生产环境超时时间前，还应使用符合上游字幕密度的样本，或全部 120 个非空位置，执行一次高计算量测试。

使用真实 JSON 文本数组时，可通过 `--text-file` 载入，并用 `--text-limit` 固定选取数量。vLLM 默认只在 Compose 私有网络中提供指标，可以读取容器地址后交给压测工具：

```bash
VLLM_IP=$(docker inspect -f \
  '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
  transflow-vllm-1)

conda run -n transflow python scripts/load_test.py \
  --base-url http://127.0.0.1:18000 \
  --dispatch-size 60 \
  --concurrency 64 \
  --requests-per-level 128 \
  --text-file /path/to/texts.json \
  --text-limit 60 \
  --languages zh,en,fr,es,ru,ar \
  --gpu-index 0 \
  --vllm-metrics-url "http://${VLLM_IP}:8000/metrics" \
  --output reports/load-real-text.json
```

测试不同在途序列时，必须同步调整 vLLM、应用调度器、批工作协程和连接数。例如 512 档使用：

```bash
VLLM_MAX_NUM_SEQS=512 \
TRANSFLOW_MAX_INFLIGHT_SEQUENCES=512 \
TRANSFLOW_BATCH_WORKERS=16 \
TRANSFLOW_MAX_CONNECTIONS=16 \
docker compose -f docker/compose.yaml up -d --force-recreate --wait
```

每个候选应独立重启并执行相同预热，避免不同的累计指标和前缀缓存状态破坏 A/B 可比性。

仅当所有已准入请求都返回 HTTP 200、输出数量完全正确、服务保持健康，并且超时、取消、OOM 和结果结构错误均为零时，该候选配置才算通过。生产参数应根据保留的测试报告手动更新；服务启动时不会自动探测显存极限。
