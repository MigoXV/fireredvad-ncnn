# FireRedVAD NCNN

独立部署的 CPU FP32 流式 VAD：使用官方 NCNN Python 绑定执行模型，
通过 NumPy、Kaldi Fbank、CMVN 和原版后处理保留音频与事件语义。
支持 Python API、离线音频推理和兼容原服务的异步 gRPC。
运行服务不需要 PyTorch、Transformers、PNNX 或原 `fireredvad` 项目。

## 安装和模型转换

```bash
poetry env use python3.10
poetry install --only main
# 仅转换模型或运行数值对齐测试时需要：
poetry install --with export
poetry run fireredvad-ncnn export \
  /workspace/opusi/fireredvad/model-bin/MigoXV/firered-vad-stream \
  model-bin/stream-vad
```

转换输入是本地 HF 格式 Stream-VAD 目录，包含 `config.json`、
`model.safetensors` 和 `cmvn.ark`。转换命令严格检查权重和结构，
使用 TorchScript → PNNX → NCNN，禁止隐式半精度转换。
输出目录必须为空，失败后应选用新的空目录重新导出。

部署只需复制以下四个文件，不需要转换生成的 `.pt` 或 Python 脚本：

- `model.ncnn.param`
- `model.ncnn.bin`
- `cmvn.ark`
- `metadata.json`

元数据记录权重摘要、工具版本、输入输出和文件 SHA-256；启动时校验文件完整性。
模型默认每块 10 帧；支持导出 1–19 帧配置，启动时分块参数必须匹配模型。
目前只支持该因果 Stream-VAD 结构（8 层、128 维缓存、19 帧历史）；不支持非流式 VAD/AED、Vulkan 或量化。

## 运行

```bash
poetry run fireredvad-ncnn infer model-bin/stream-vad audio.wav
poetry run fireredvad-ncnn serve model-bin/stream-vad \
  --host 0.0.0.0 --port 50051 --num-threads 4 \
  --stream-chunk-frame 10 --min-silence-frame 20 \
  --engine-session-queue-capacity 2
poetry run fireredvad-ncnn serve --help
```

入口是 `fireredvad_ncnn.commands.app`。原客户端可连接相同 gRPC 服务名
`ux_vad.UxVoiceActivityDetector`，调用 `Detect` 和 `StreamingDetect`。
保留请求级后处理配置、`--endpoint-only`、健康检查、有界队列、会话顺序、
取消、超时和进程退出清理。`processed_time` 是最终归属水位，不是累计输入进度。
不支持的设备、精度或 runner 会报错；不会静默转回 Torch。

```python
import numpy as np
from fireredvad_ncnn.inferencers.inferencer import Inferencer

engine = Inferencer("model-bin/stream-vad", ncpu=4)
try:
    segments = engine.infer(np.zeros(16000, dtype=np.int16), 16000)
    # 流式输入为 (单声道 int16 PCM, 采样率, 是否最后一块)。每次调用拥有独立会话。
    events = list(engine.infer_stream([
        (np.zeros(1600, dtype=np.int16), 16000, False),
        (np.zeros(1600, dtype=np.int16), 16000, True),
    ]))
finally:
    engine.close()
```

原始音频按 int16 PCM 幅值解释；浮点输入也使用 PCM 幅值，不是 [-1, 1] 归一化波形。
每个会话拥有独立 Fbank、重采样器、后处理器和八层模型缓存。
NCNN 每次执行创建 Extractor，复制返回值；候选缓存只在请求仍有效时提交。
调度批次内逐路执行，不把多路音频拼接为一条时间序列。
离线推理采用有状态分块，在 `chunk_max_frame` 边界重置，保持原版离线语义。

## 延迟配置

默认 10 帧对应 100 ms 推进量，25 ms 特征窗意味着首次模型块需要 115 ms 音频。
默认平滑 5 帧、语音阈值 0.5、回补 5 帧、语音确认 8 帧、结束静音 20 帧。
这些后处理值来自官方配置类；100 ms 分块来自原项目，并非官方固定的总延迟。
原项目 CLI 的结束静音默认是 40 帧，因此对比命令显式指定双方为 20 帧。

服务沿用原版双向流处理方式：默认积累 2 个待处理音频块后读取响应，
因此真实发送时响应可见时间还受 API 缓冲、客户端发送节奏和排队影响。
固定相同配置不代表实测延迟相等；基准单独保留实时发送的首响应和 EOF 排空耗时。
2 块窗口实测首响应约 302 ms：第一块 100 ms 音频不足以产生首次特征，
第二块结果在收到第三块时被读取，不能将 2 块窗口直接等同于 200 ms 首响应。

## 测试和短时基准

```bash
poetry install --with export
poetry run pytest -q
poetry run fireredvad-ncnn benchmark model-bin/stream-vad \
  --reference-project /workspace/opusi/fireredvad \
  --reference-model /workspace/opusi/fireredvad/model-bin/MigoXV/firered-vad-stream \
  --wav-dir /workspace/opusi/fireredvad/data-bin/0713 \
  --output benchmark-results/run-001
```

基准需要 dev 依赖、已可运行的原项目 Poetry 环境，以及至少 8 个可用物理核心。
双方固定四个相同物理 CPU 核心、4 个计算线程、2 块待处理窗口；客户端使用另外四核。
每档交替启动后端，计时前加载音频、转为 16 kHz 并预热。
输入固定为 10 秒，按 100 ms 音频块不限速发送，统计包括 gRPC、IPC、特征提取、模型及后处理。
该结果反映当前机器与线程配置，不是整机满核极限吞吐。

仅测 1、4、8、16、32 路：每档先用 2 秒音频短测，任一路低于 2× 实时，
就跳过该后端该档正式测试和后续更高并发；另一后端独立继续。
通过短测后测 3 轮，正式轮次若失败或跌破阈值也立即停止。
输出 `raw.jsonl`、`summary.json`、`environment.json`、服务日志和固定输入 WAV。
总吞吐为成功音频秒数/整轮墙钟秒数；每路倍数为该路音频时长/完成耗时。
报告失败数，不把失败隐藏在吞吐平均值里。

模型相关测试在缺少本地模型时跳过；可通过 `NCNN_TEST_MODEL`、
`VAD_REFERENCE_MODEL` 和 `VAD_REFERENCE_PROJECT` 指定验证资源。
数值测试使用 `atol=1e-4, rtol=1e-4`，覆盖概率、缓存、短块继续推理和真实音频。
服务测试比较原客户端协议输出，覆盖重采样、静音、短音频、EOF、水位和故障退出。

代码来源见 [NOTICE.md](NOTICE.md)。模型与大体积临时产物不纳入 Git；基准报告单独保存。

当前 2 块窗口结果见 [延迟与吞吐复测](reports/cpu-20261011-window2/README.md)：
首响应从约 900 ms 降至约 302 ms，包含五档吞吐和全部原始数据。
此前 [8 块窗口报告](reports/cpu-20261011/README.md) 保留为历史对照。
