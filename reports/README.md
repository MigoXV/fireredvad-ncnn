# 验证与性能报告索引

本目录保留历史验收结论、关键性能汇总和量化方法依据。过程日志、Docker 完整 inspect、逐文件推理详情和重复模型产物已清理；完整记录可从 Git 历史恢复。历史部署地址、PID 和健康状态不代表当前在线状态。

## 当前冒烟（2026-11-09）

代码版本：`36fb34f`。使用本地 `model-bin/stream-vad` 和真实语音 `model-bin/speech.wav`，通过 Poetry 执行：

```bash
poetry run pytest -q tests/test_service_integration.py tests/test_cli.py tests/test_engine_config.py
```

结果：**11 passed**。覆盖新实例启动就绪、离线 Detect、StreamingDetect、多采样率/短音频/空输入与 native API 对齐、请求级阈值、单进程无子进程、正常退出、无需 Torch 等研究依赖、关闭超时退出及 CLI 配置。未重跑全量精度和性能测试，也未验证历史部署容器。

## 建议阅读

- [单进程迁移结果](singleprocess-20261014/result.json)及[性能对照](singleprocess-20261014/comparison.json)：旧 IPC 与进程内桥接的对比；历史测试另有 3 项已复现的 8 kHz 参考边界差异，详见结果中的限制。
- [真实 ARM64 验收](arm64-20261014/validation.json)：Kunpeng 920 的原生构建、数值与服务验证。
- [原生迁移报告](native-20261013/README.md)：原 Python NCNN 与 C++ 引擎的历史对照。
- [INT8 量化实验](quantization-20261020/README.md)：保留方法、精度与吞吐汇总，以及正式基准原始轮次和环境。constant-A INT8 在该实验中有效；直接 Gemm INT8 精度失败，不能作为默认部署依据。

其余日期目录仅保留汇总或结论；`cpu-20261011*` 是早期 Python 引擎结果。不同硬件、输入和服务架构的数据不能直接混作当前性能。
