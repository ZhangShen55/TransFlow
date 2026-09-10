# 负载测试

容量测试必须在指定且无其他负载的 GPU 上运行。使用待测的调度分片大小启动服务，并在报告中记录相同数值：

```bash
TRANSFLOW_MODEL_PATH=/var/model_llm/Hy-MT2-1.8B \
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

仅当所有已准入请求都返回 HTTP 200、输出数量完全正确、服务保持健康，并且超时、取消、OOM 和结果结构错误均为零时，该候选配置才算通过。生产参数应根据保留的测试报告手动更新；服务启动时不会自动探测显存极限。
