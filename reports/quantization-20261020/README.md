> 归档说明（2026-11-09）：本目录是历史实验记录，不代表当前工作区的验证结果。报告已精简；过程日志、逐文件结果、容器快照及模型副本已移除，文中相关文件引用可从 Git 历史恢复。保留结论和汇总；量化正式基准另保留原始轮次与环境。历史命令及部署状态以记录日期为准。

# Kunpeng 920 上 FireRedVAD NCNN INT8 实测

推荐使用显式的 `constant-a-int8` 量化产物：同图对照吞吐提升约 18%–21%，权重文件缩小 73.4%，两份人工标注测试集的总体分段 F1 最大降低 0.019 个百分点。本目录记录 2026-10-20 的真实 ARM64 量化实验。原始模型、FP32 对齐、公开人工标注、正式吞吐原始记录和失败变体均予保留。此次仅新增研究脚本、验证和报告，未修改仓库中用户已有的运行时、配置、README 或未提交改动。

## 基线与量化方法

- 唯一语义基准：`/root/repositories/fireredvad`，revision `05107f7b2df88edb5a8242d678b48b747c181fb5`。直接导入其中 `module_firered_vad_stream.py`，以 CPU Torch 2.9.1 加载原始 safetensors；未重新实现一份模型冒充基准。
- 模型：`MigoXV/firered-vad-stream`，HF revision `b7f3207233ae36021bce31c5561549edc80e3325`，权重 SHA256 `715b19620ce21e885dc7c3c399231111c761966394ea34f6bb29f1731ed95070`。
- 原始 NCNN FP32 产物：`model-bin/stream-vad`，10 帧一块，8 个 `[128,19]` 历史 cache。输入特征、cache 和输出仍是 FP32，INT8 指内部矩阵计算与权重。
- 使用仓库原生引擎所链接的 NCNN `20260526`，编译其**未修改**的 `ncnn2table` / `ncnn2int8`。工具 hash 见转换产物中的 `conversion.json`，工具源码 hash 和编译设置见本目录 `audit.json`。

| 变体 | 实现 | 权重 bin 字节 | 相比 FP32 缩小 |
|---|---|---:|---:|
| `stream-vad` | 原图 FP32 Gemm | 2,271,896 | — |
| `gemm-int8` | 官方直接转换；constant-B 的整个权重矩阵一个 scale，动态激活量化 | 591,072 | 73.98% |
| `ip-fp32` | 数学等价 Gemm → InnerProduct，未量化 | 2,271,852 | 0.002% |
| `ip-int8-kl` | InnerProduct 逐输出通道权重 + 真实数据 KL 静态激活量化 | 609,208 | 73.19% |
| `constant-a-fp32` | 将 `X @ W.T + b` 等价改写为 `(W @ X.T + b[:,None]).T` | 2,271,896 | — |
| `constant-a-int8` | 上述改图后官方动态 INT8；constant-A 权重逐行 scale | 604,316 | 73.40% |

原图的算子是 Gemm 和 depthwise Conv1D；该版本 `ncnn2table` 不校准这两个算子的动态输入。因此直接原图调用工具会产生空静态 table，不能称为静态 KL 校准。日志保存在 `gemm-int8/ncnn2table.log`。为实际验证校准量化，增加了严格检查二进制布局的 InnerProduct 等价改图，真实 table 有 18 层。再以 constant-A 等价改图探索逐通道动态 INT8。未支持的参数、权重格式或未消费的二进制字节会直接报错。

所有 INT8 图均只量化 18 个线性层；8 个 depthwise Conv1D 仍为浮点计算。官方 `ncnn2int8` 默认将未量化权重以 FP16 存储，加载后由当前禁用 FP16 arithmetic 的运行时执行。这里没有宣称整网所有算子都为 INT8。

既有 `frv_build_info()` 中 `cpu=fp32` 是硬编码的引擎构建信息；此次没有改动该接口。实际模型精度应查看 `metadata.json.quantization`、转换日志以及 `.param` 的 INT8 参数，不能只凭这个既有字符串判断量化是否执行。

## 标注、校准与精度口径

1. [TEN 官方人工标注测试集](https://github.com/TEN-framework/ten-vad/tree/d68ab71c8d5fbd8f98e97ae7c9986af76b5d72b1/testset)：固定 revision `d68ab71c8d5fbd8f98e97ae7c9986af76b5d72b1`。下载全部 30 条，仅约 8.4 MB。**01–10 仅用于校准**，95.605 秒；**11–30 为测试**，166.711 秒。该划分在推理前固定；未用测试标签调阈值。官方未提供原始说话人/录音分组信息，这里保证文件不重叠，不额外声称说话人独立。
2. [VoxConverse 官方 RTTM](https://github.com/joonson/voxconverse/tree/24bf60be297701cd7e4ef18550c6d390c1b87365)：独立测试 `msbyq/dohag/neiye/xkmqx`，181.120 秒，涵盖多说话人与重叠。多个说话人的语音时间取并集作为 VAD 真值；没有将重叠时间重复计入。音频来自锁定 revision 的公开 HF 镜像，来源、许可、音频和 RTTM SHA256 在数据 manifest 中。两条 calibration 录音没有用于本实验拟合。

校准逐条遍历完整录音，每条均匀选择 16 块，捕获 160 组真实 10 帧特征（合计约 16 秒特征）和 8 组真实前序 cache。只有每条录音的第一块 cache 为全零，其余均来自该 FP32 模型的真实前向输出；未用随机数或常量伪造校准。`calibration/manifest.json` 记录文件、块编号、每个输入的 SHA256 和范围。

原始概率按 0.5 固定阈值计分，使用 10 ms 特征中心时间 `i*0.01+0.0125`。生产 native 离线分段在 10 ms 网格中点计分，**无 collar**，后处理参数完全一致：平滑 5 帧、起点补 5 帧、最短语音 8 帧、结束静音 20 帧、最长段 2000 帧。两种口径分别报告；后处理会延展语音，不能把原始概率和最终分段的 FA 混为一谈。

`MISS=FN/(TP+FN)`，`FA=FP/(FP+TN)`，precision、recall 和 F1 都是帧计数的 micro 指标。准入门槛预先定为总体 F1 降低不超过 0.5 个百分点、无运行错误；是否加速另以正式吞吐判断。

### 生产 native 分段质量

| 数据集 | 变体 | Precision % | Recall % | F1 % | MISS % | FA % | F1 相对 FP32 变化 pp |
|---|---|---:|---:|---:|---:|---:|---:|
| TEN | FP32 | 88.1659 | 96.3133 | 92.0597 | 3.6867 | 37.8506 | — |
| TEN | 直接 Gemm INT8 | 86.3818 | 20.8323 | 33.5690 | 79.1677 | 9.6158 | -58.4907 |
| TEN | IP INT8 KL | 88.1577 | 96.2972 | 92.0479 | 3.7028 | 37.8741 | -0.0118 |
| TEN | constant-A INT8 | 88.1447 | 96.2972 | 92.0408 | 3.7028 | 37.9213 | -0.0189 |
| VoxConverse | FP32 | 98.5859 | 96.5214 | 97.5428 | 3.4786 | 25.8696 | — |
| VoxConverse | 直接 Gemm INT8 | 98.8904 | 37.3277 | 54.1976 | 62.6723 | 7.8261 | -43.3451 |
| VoxConverse | IP INT8 KL | 98.5809 | 96.5796 | 97.5700 | 3.4204 | 25.9783 | +0.0272 |
| VoxConverse | constant-A INT8 | 98.5747 | 96.5505 | 97.5521 | 3.4495 | 26.0870 | +0.0093 |

FP32 原图及两种等价 FP32 改图在全部测试文件上，原始阈值分类和最终分段指标逐项一致，均与原仓库 Torch 语义基准对齐。原 NCNN 对 Torch 的概率最大绝对误差不超过 `4.77e-7`，IP FP32 改图不超过 `1.32e-6`。

原始概率 F1（TEN / VoxConverse）分别为：FP32 `92.6971% / 93.3692%`，IP INT8 KL `92.7448% / 93.4067%`，constant-A INT8 `92.6902% / 93.3609%`；直接 Gemm INT8 只有 `29.6392% / 41.8816%`，**精度失败，不能部署为默认**。逐文件原始计数、概率误差与片段均保留。IP INT8 在 TEN 最差单条分段 F1 降低约 0.522 个百分点，所以总体近似不变不代表每条音频都不退化。

所选 constant-A INT8 最差单条分段 F1 变化为 TEN `-0.4174 pp`（`testset-audio-28`）、VoxConverse `-0.0651 pp`（`dohag`）。其相对原 Torch 的概率最大绝对误差分别为 `0.09159 / 0.04349`，按文件音频长度加权的平均绝对误差约 `0.00387 / 0.00216`。

## 正式性能方法

硬件为真实 `aarch64 / Kunpeng 920 / 64 物理核`，不是 QEMU。协调者停掉原有 NPU 服务及 CPU VAD 服务后发布 `device-ready`；每次正式矩阵持有统一 `device.lock`。其他任务的编译/工作固定 0–31 核；4 个原生 worker 分别固定 48、49、50、51 核，每 worker NCNN 1 线程；调用与资源采样线程固定 52–55 核。

固定同一 20 条测试录音，重复 4 轮形成每轮 80 请求、666.845 音频秒；没有拼接录音。并发档 `1/2/4/8/16/32`，每个模型与并发档先预热 4 条，再正式运行 3 轮；模型顺序按并发档交错正反。闭环补齐请求，最多维持指定并发数。

吞吐倍数为 **成功输入音频总秒数 / 整轮 makespan**。延迟是完整 PCM 提交到完整 native 结果，包含复制、排队、Fbank/CMVN、模型和后处理；不含读盘、模型加载和网络。这是原生端到端运行时测量，**不是 gRPC 网络吞吐**。P50/P95/P99 汇总该并发档三轮共 240 个请求，各条真实录音长度不同。

以下结果由原始 JSON 自动汇总；CPU、RSS、线程/FD、分阶段计时和轮间波动在原始记录中。RSS 是同进程的进程峰值，包含库和分配器缓存，不等于单份模型占用。

### 原生吞吐：每秒可处理音频秒数

| 并发 | 原图 FP32 × | 直接 Gemm INT8 ×（精度失败） | IP FP32 × | IP INT8 KL × | constant-A FP32 × | constant-A INT8 × |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.14 | 59.85 | 40.83 | 38.15 | 51.20 | 61.58 |
| 2 | 96.92 | 117.57 | 78.26 | 74.74 | 99.37 | 119.51 |
| 4 | 177.24 | 217.52 | 140.74 | 141.89 | 185.37 | 222.40 |
| 8 | 181.21 | 222.45 | 144.17 | 145.96 | 194.23 | 229.36 |
| 16 | 180.94 | 223.48 | 143.74 | 145.43 | 191.55 | 228.54 |
| 32 | 183.30 | 223.86 | 144.37 | 146.64 | 191.45 | 230.95 |

同一 constant-A 图的 INT8 相对 FP32 吞吐提升为 **18.09%–20.63%**；若把等价改图的收益也计入，对原图 FP32 提升为 **22.81%–26.58%**。两种归因分开计算：原图对照来自第一串行矩阵，constant-A FP32/INT8 来自第二串行矩阵，硬件、输入、CPU预算、并发和请求数均一致。

### 原生完成延迟：毫秒，三轮请求合并后分位数

| 并发 | 原图 FP32 P50 | P95 | P99 | constant-A INT8 P50 | P95 | P99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 175.5 | 312.3 | 361.5 | 143.1 | 231.4 | 256.3 |
| 2 | 180.1 | 296.6 | 322.3 | 146.1 | 240.4 | 260.0 |
| 4 | 189.0 | 317.7 | 347.4 | 149.7 | 251.5 | 276.6 |
| 8 | 363.3 | 614.4 | 689.2 | 284.1 | 489.6 | 549.1 |
| 16 | 667.7 | 1260.0 | 1385.2 | 524.3 | 1010.0 | 1097.6 |
| 32 | 1248.0 | 1895.3 | 2678.7 | 988.3 | 1497.1 | 2128.4 |

32 并发平均服务计算用量约 `3.86` 核（FP32）与 `3.81` 核（constant-A INT8）；采样进程峰值 RSS 分别约 `113.6` / `106.5` MiB。RSS包含进程库、四份模型及分配器缓存，不能按此差值宣称单份模型的常驻内存节省。

两阶段共 **108 个正式轮次、8640 个请求，零失败**；每轮结束的 `pending/sessions/inflight_bytes` 均为零。constant-A INT8 的 18 个轮次吞吐波动均小于 1%，完整 min/max 见 JSON。第一阶段保留了原图单并发的一个慢轮次 `44.63×`，以及直接 Gemm INT8 的 32 并发 `126.59×/223.86×/226.92×` 和 IP INT8 的 `131.71×/146.64×/146.98×`；没有删除慢轮次。这里是共享服务器上经服务停机和核绑定隔离的测量，不额外声称操作系统级独占物理主机。

静态 IP INT8 保留精度、压缩模型，但其吞吐低于原图 FP32，不能作为加速默认；直接 Gemm INT8 稍快却严重丢失召回，同样不能使用。综合这些证据，推荐**显式选择 `constant-a-int8` 产物**，并在实际业务语料上继续验收。

### 真实 Python gRPC Detect 端到端基准

继续使用相同数据、4 个原生单线程 worker、相同 CPU 隔离和统一设备锁。这里计入 loopback HTTP/2、protobuf、Python async gRPC 接口、native 队列、DSP、NCNN 和后处理。连接建立、模型加载与读盘在预热阶段完成。每个并发档使用全新服务进程，预热 4 请求后测试 3 轮；每轮 80 条、666.845 音频秒。

| 并发 | 原图 FP32 × | constant-A INT8 × | 吞吐变化 |
|---:|---:|---:|---:|
| 1 | 49.28 | 60.29 | +22.34% |
| 2 | 95.11 | 117.11 | +23.13% |
| 4 | 165.74 | 205.63 | +24.07% |
| 8 | 171.65 | 214.13 | +24.75% |
| 16 | 171.11 | 214.14 | +25.15% |
| 32 | 171.54 | 214.54 | +25.07% |

| 并发 | FP32 P50 ms | P95 ms | P99 ms | INT8 P50 ms | P95 ms | P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 178.3 | 286.7 | 318.4 | 145.8 | 233.5 | 259.4 |
| 2 | 183.4 | 301.7 | 330.2 | 149.1 | 245.3 | 265.0 |
| 4 | 200.7 | 349.5 | 369.9 | 160.8 | 284.0 | 298.7 |
| 8 | 381.2 | 646.0 | 727.5 | 302.5 | 516.5 | 587.3 |
| 16 | 727.5 | 1335.0 | 1498.6 | 579.9 | 1073.4 | 1207.8 |
| 32 | 1369.4 | 2114.5 | 2915.4 | 1093.0 | 1695.7 | 2344.4 |

32 并发的服务端平均 CPU 用量约为 FP32 `3.78` 核、INT8 `3.76` 核；采样峰值 RSS 为 `154.0` / `147.3` MiB。完整资源、逐请求延迟与命令见 `benchmark-grpc/`。

gRPC 共 **36 轮、2880 请求，零拒绝、超时或内部错误**；12 个服务均正常退出、无子进程、监听端口可重新绑定。每个服务的最终 native `sessions/pending/inflight_bytes` 均为零，完整日志与 `lifecycle-with-native-metrics.json` 保留在本目录。流式 gRPC 的协议、结果和退出已另经真实 INT8 集成测试验证；此表的性能协议为一元 `Detect`，不把它称为流式 RPC 吞吐。

脚本溯源限制：[正式运行环境记录](benchmark-grpc/environment.json)中的 `scripts/quantization_research.py` SHA256 为 `0fa07c627c5377e531b926463768d2c1b67a4b981650fbf880cb2dd4a43a10c9`，[交付审计记录](audit.json)中的 SHA256 为 `5e4dfb71eca8530ddb475f64952d32faed4e65febff37359216d702de2f750ef`。两者不同，且未保存正式运行时的整份脚本快照，因此不宣称两版脚本字节相同。本节指标来自保存的 [raw.jsonl](benchmark-grpc/raw.jsonl)；交付脚本提供复现入口，不能作为当时完整源码的逐字节快照。

## 路径与复现

- 模型变体：`/home/model-bin/MigoXV/firered-vad-stream-ncnn/`。
- TEN 数据与 provenance：`/home/ascend-audio-eval-20261020/data/ten-vad/`。
- VoxConverse 数据与 manifest：`/home/data-bin/audio-eval-20261020/voxconverse/`。
- 校准张量、逐帧概率、全量运行日志和原始结果：`/home/ascend-audio-eval-20261020/ncnn-quant/`。
- 研究环境：`/home/ascend-audio-eval-20261020/virtualenvs/ascend-audio-eval-83_NCwnu-py3.10`，由 Poetry 管理。Torch 通过只读 `.pth` 借用已经部署的 CPU Torch，未复制大包到根盘。

```bash
cd /root/repositories/fireredvad-ncnn
export VIRTUAL_ENV=/home/ascend-audio-eval-20261020/virtualenvs/ascend-audio-eval-83_NCwnu-py3.10
export PYTHONPATH="$PWD/src"
export FIREREDVAD_LIBRARY="$PWD/build/native/libfireredvad.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VAD_FP32_MODEL="$PWD/model-bin/stream-vad"
export VAD_DATASET_DIR=/home/new-run/data/ten-vad
export VAD_CALIBRATION_DIR=/home/new-run/calibration
export NCNN_QUANTIZATION_TOOLS=/home/new-run/tools

# 在新目录复现时先构建 native 运行时，-B 应放到 /home；该次实验复用已有 build/native。
bash scripts/build_quantization_tools.sh "$PWD/build/native" "$NCNN_QUANTIZATION_TOOLS"

# fetch/rewrite/calibration/quantize/accuracy/benchmark 的输出使用全新路径；不会覆盖已有结果。
poetry run python scripts/quantization_research.py --help
poetry run python scripts/quantization_research.py fetch --output /home/new-run/data/ten-vad
poetry run python scripts/quantization_research.py calibration --output /home/new-run/calibration
poetry run python scripts/quantization_research.py rewrite --strategy innerproduct --output /home/new-run/ip-fp32
poetry run python scripts/quantization_research.py quantize --source /home/new-run/ip-fp32 --output /home/new-run/ip-int8-kl
poetry run python scripts/quantization_research.py rewrite --strategy constant-a --output /home/new-run/constant-a-fp32
poetry run python scripts/quantization_research.py quantize --source /home/new-run/constant-a-fp32 --output /home/new-run/constant-a-int8

poetry run python scripts/quantization_research.py accuracy \
  --model "$VAD_FP32_MODEL" --model /home/new-run/ip-int8-kl --model /home/new-run/constant-a-int8 \
  --source-repo /root/repositories/fireredvad --weights /home/model-bin/MigoXV/firered-vad-stream \
  --output /home/new-run/accuracy

# 必须先由协调者完成服务隔离，再进行正式性能测试；脚本检查 ready 并自动 flock。
taskset -c 52-55 poetry run python scripts/quantization_research.py benchmark \
  --model "$VAD_FP32_MODEL" --model /home/new-run/constant-a-fp32 --model /home/new-run/constant-a-int8 \
  --cpu-affinity 52,53,54,55 --worker-cpu-affinity 48,49,50,51 --workers 4 \
  --output /home/new-run/benchmark
taskset -c 52-55 poetry run python scripts/quantization_research.py grpc-benchmark \
  --model "$VAD_FP32_MODEL" --model /home/new-run/constant-a-int8 \
  --client-cpus 52,53,54,55 --server-cpus 48,49,50,51 --workers 4 \
  --output /home/new-run/benchmark-grpc
```

部署沿用现有 C ABI 原生端到端执行与 Python gRPC 接口，显式传入量化模型路径即可，外部特征/cache 的 `dtype` 仍为 FP32。未改协议、后处理阈值或全局默认模型路径。

```bash
poetry run python -m fireredvad_ncnn.commands.app infer \
  /home/model-bin/MigoXV/firered-vad-stream-ncnn/constant-a-int8 \
  "$VAD_DATASET_DIR/testset-audio-11.wav" --num-threads 4
poetry run python -m fireredvad_ncnn.commands.app serve \
  /home/model-bin/MigoXV/firered-vad-stream-ncnn/constant-a-int8 \
  --host 127.0.0.1 --port 50051 --num-threads 4 --engine-workers 4 --threads-per-worker 1
```

新增验证覆盖标签缺口/重叠/非法值、RTTM 重叠说话人合并、FP32 二进制布局与错误拒绝、两种等价变换、真实量化模型 8 路流式状态隔离、gRPC 离线/流式与 native 对齐、就绪和退出后端口释放。运行命令：

```bash
export NCNN_INT8_TEST_MODEL=/home/model-bin/MigoXV/firered-vad-stream-ncnn/constant-a-int8
export NCNN_INT8_TEST_WAV="$VAD_DATASET_DIR/testset-audio-11.wav"
taskset -c 0-3 poetry run pytest -q tests/test_quantization_research.py tests/test_quantized_runtime_integration.py
```

constant-A 产物的全部 8 项验证通过，静态 IP 产物也通过相同 native/gRPC 验证。没有对用户原有未提交的整个工作区执行重置、覆盖或提交。
