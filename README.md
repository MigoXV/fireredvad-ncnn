# FireRedVAD NCNN 原生引擎

C++17 实现的 CPU FP32 FireRedVAD 流式引擎，对外提供纯 C ABI。
SO、原生 CLI、Python API 和 Python gRPC 服务使用同一套音频处理与推理实现。
NCNN、重采样、Fbank、CMVN、模型缓存、后处理和会话调度都在原生侧运行。
原生库与 CLI 无需 Python；Python 包绑定的是 FireRedVAD 引擎。

第一版支持 Linux x86_64、ARM64。默认将 NCNN、Fbank 和 KissFFT 静态链接进
`libfireredvad.so`，libsoxr 与系统运行库动态链接。可额外静态链接 libsoxr；
完全静态可执行文件和原生 gRPC 不在本版范围内。

## 原生构建与部署

需要 C/C++ 编译器、CMake 3.20–3.x、Ninja 和 OpenMP。首次配置下载固定版本的源码，
并校验 SHA-256；不下载模型。构建过程中不使用 NCNN Python 绑定。

```bash
cmake --preset linux-x86_64
cmake --build --preset linux-x86_64
ctest --test-dir build/x86_64 --output-on-failure
cmake --install build/x86_64 --prefix "$PWD/build/install-x86_64"

# 需安装 aarch64-linux-gnu-gcc / g++ 交叉工具链
cmake --preset linux-arm64
cmake --build --preset linux-arm64
cmake --install build/arm64 --prefix "$PWD/build/install-arm64"
```

ARM64 设备本机编译时使用 `cmake -S . -B build/native -G Ninja`，无需交叉工具链。
交叉编译后仍需在目标设备验证；QEMU 可用于功能冒烟，不能用于性能验收。

安装目录包含 `include/fireredvad.h`、SO、静态库及依赖、CLI、CMake package 和许可证。
复制安装目录及模型目录即可部署，系统仍需匹配的 libc、libstdc++ 和 OpenMP 运行库。
SO 导出符号仅限 `frv_*` C ABI。系统运行库兼容性由构建工具链决定。

构建选项：

- `FRV_BUILD_SHARED`、`FRV_BUILD_STATIC`、`FRV_BUILD_CLI`、`FRV_BUILD_TESTS`：默认开启。
- `FRV_STATIC_DEPS=ON`：默认静态链接 NCNN/Fbank；关闭后需要部署对应动态依赖。
- `FRV_STATIC_SOXR=ON`：额外静态链接 libsoxr，安装原生引擎对象、soxr 源码及构建材料。
- `FRV_WHEEL_LIB_DIR`：将 SO 和 libsoxr 复制到指定 Python 包资源目录。
- `FETCHCONTENT_SOURCE_DIR_NCNN/FBANK/SOXR/JSON/SHA/KISSFFT`：指定已解压的本地依赖源码，支持离线构建。

libsoxr 使用 LGPL-2.1；默认以可替换的动态库交付。静态分发应同时保留安装的
`share/fireredvad/relink`、soxr 源码、依赖归档及工具链说明，供使用方重新链接。
模型不嵌入二进制，也不随 wheel 分发。

## 模型与 Python 开发环境

项目使用 Poetry。开发和测试命令通过 `poetry run` 执行。

```bash
poetry env use python3.10
poetry install --only main
cmake --preset linux-x86_64 \
  -DFRV_WHEEL_LIB_DIR="$PWD/src/fireredvad_ncnn/lib"
cmake --build --preset linux-x86_64

# 转换本地 HF Stream-VAD 权重需要 export 组；验证参考实现需要 dev 依赖
poetry install --with export
poetry run fireredvad-ncnn export /path/to/hf-stream-vad model-bin/stream-vad
```

转换输入包含 `config.json`、`model.safetensors` 和 `cmvn.ark`。
转换采用 TorchScript → PNNX → NCNN FP32，输出目录必须为空。
部署只需四个文件：`model.ncnn.param`、`model.ncnn.bin`、`cmvn.ark`、`metadata.json`。
原生启动检查 SHA-256、格式、维度和分块配置；保留现有 format_version=1 模型包。
仅支持 8 层、128 维缓存、19 帧历史、80 维特征的因果 Stream-VAD。
导出和推理分块支持 1–19 帧，默认 10 帧，两者必须一致。

Python 通过 `ctypes.CDLL` 调用 C ABI，计算及等待调用释放 GIL。
优先加载包内 `lib/libfireredvad.so`；开发环境也可加载构建目录。
`FIREREDVAD_LIBRARY=/absolute/path/libfireredvad.so` 可指定独立安装的引擎，
指定路径错误时明确失败。运行不需要 Torch、PNNX、NCNN Python、SciPy、
Kaldi Python 或 soxr Python；它们仅用于导出、参考测试或基准工具。

## CLI、Python API 与 gRPC

```bash
# 完全独立于 Python
build/x86_64/fireredvad infer model-bin/stream-vad audio.wav
build/x86_64/fireredvad stream model-bin/stream-vad audio.wav --chunk-ms 100
build/x86_64/fireredvad stream model-bin/stream-vad - --raw-rate 16000 < audio.pcm

# Python 入口
poetry run fireredvad-ncnn infer model-bin/stream-vad audio.wav
poetry run fireredvad-ncnn serve model-bin/stream-vad \
  --host 0.0.0.0 --port 50051 --num-threads 4 \
  --engine-workers 4 --threads-per-worker 1
```

原生 CLI 接收单声道 PCM16 RIFF WAV；`stream` 另支持显式采样率的原始 PCM16。
离线输出 JSON，流式每个输入块输出一行 JSONL；日志写 stderr，失败返回非零。
`--help` 列出线程、分块和后处理参数，`--endpoint-only` 只输出闭合片段。

```python
import numpy as np
from fireredvad_ncnn.inferencers.inferencer import Inferencer

with Inferencer("model-bin/stream-vad", ncpu=4) as engine:
    segments = engine.infer(np.zeros(16000, dtype=np.int16), 16000)
    events = list(engine.infer_stream([
        (np.zeros(1600, dtype=np.int16), 16000, False),
        (np.zeros(1600, dtype=np.int16), 16000, True),
    ]))
```

保留 `infer()`、`infer_stream()`、`close()` 和结果形态，每次调用拥有独立会话。
音频为单声道 int16 或浮点 PCM 幅值，浮点不是 [-1,1] 归一化波形。
流式会话采样率不能变化，最终块之后不能继续输入。

gRPC 保留服务名 `ux_vad.UxVoiceActivityDetector`、`Detect`、`StreamingDetect`、
请求级后处理配置、健康检查和原有响应语义。Python 负责协议与进程内异步适配，原生引擎拥有全部计算和会话状态。
服务只运行一个进程：一个桥接线程持有原生引擎，C++ worker 线程执行计算，
结果通过线程安全回调交给 asyncio，不使用 ZMQ、子进程或 IPC 序列化。
`processed_time` 是不会再改变音频归属的最终水位；不是累计输入进度。
服务保留两块响应读取窗口。默认 10 帧推进 100 ms，首次特征块需要 115 ms 音频；
可见响应延迟还取决于客户端节奏、API 窗口及排队。

## 多核、容量与 C ABI

`--num-threads` 是计算 CPU 预算；`engine_workers × threads_per_worker` 必须不超过预算。
默认 `4 / 1 / 4`，优先保持单流行为；多流可使用 `4 / 4 / 1`，不自动切换策略。
每个 worker 独立持有 NCNN 模型实例，因此增加 worker 数也增加模型副本内存。
模型每块执行后重新调度，长音频让出计算资源；同会话严格 FIFO，跨会话公平执行。
原生引擎按块调度，不等待凑批，也不拼接多路音频。服务参数仅保留实际生效的配置；
已删除 `--batch-size`、`--batch-wait-ms`、`--workers` 和 `--chunk-size` 兼容入口。

原生限制会话数、每会话等待队列、未释放请求数和输入/结果内存预算。
每会话队列容量指一个活跃请求之外的等待数量。结果完成后仍计入容量，
直到调用方消费并释放；模型副本和系统运行库内存不属于该请求预算，另行观察 RSS。
Python 适配层的排队、执行及未消费完成结果也受数量和字节容量限制；
取消请求的容量在原生完成被回收后才释放。会话清理使用独立的有界控制通道。

取消和 deadline 使用单调时钟；排队请求不执行，运行中请求在原生步骤完成后丢弃结果。
失败、取消或超时使对应流式会话失效，后续输入必须新建会话。
每次服务启动创建独立引擎，旧会话不能跨实例使用。关闭先停止接收，再销毁引擎并
等待正在执行的原生步骤及桥接线程结束。超过关闭宽限期时服务非零退出，
由容器重启策略恢复；原生崩溃会影响整个服务进程。

`src/native/include/fireredvad.h` 是公共接口：

- `frv_engine_create/destroy`、`frv_session_open/close` 管理生命周期。
- `frv_submit/poll/cancel` 提供异步提交、完成结果和取消；同步 `frv_detect/stream_push` 复用同一调度路径。
- `frv_get_metrics` 导出请求、容量、模型块数及预处理/推理/后处理累计耗时。
- 配置通过 init 函数初始化，携带大小与 ABI 版本；异常在边界转成状态码。
- 提交成功前复制输入；结果由引擎分配，调用方使用 `frv_result_free` 释放。结果可在引擎销毁后释放。
- `poll` 和同步接口返回 0 表示取得结果，执行是否成功还需检查 `result->status`；错误详情在 `result->message`。
- 会话序号从 1 开始，仅成功接收才推进；句柄属于创建它的引擎。销毁前调用方必须停止并发使用句柄。

纯 C 示例及独立消费项目位于 `src/native/examples`：

```bash
cmake -S src/native/examples -B build/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/install-x86_64" \
  -DFRV_EXAMPLE_STATIC=ON
cmake --build build/consumer
build/consumer/detect model-bin/stream-vad
```

消费项目支持 `FireRedVAD::shared` 和 `FireRedVAD::static`。
应用源码可以是纯 C；静态链接时最终链接步骤需包含 C++ 和 OpenMP 运行库，由 CMake target 自动传递。

## Wheel、测试与基准

wheel 使用项目 PEP 517 后端，先构建原生库，再使用 Poetry 打包：

```bash
poetry run python -m build --wheel
FIREREDVAD_BUILD_ARCH=aarch64 poetry run python -m build --wheel
```

第二条命令在 x86_64 上需要 ARM64 交叉工具链。
产物为 `py3-none-linux_x86_64` 或 `py3-none-linux_aarch64`，包含引擎 SO、libsoxr 和许可证。
它们带平台与系统运行库要求；不声明 manylinux 兼容。源码包包含原生源码与构建入口。

```bash
# 显式下载固定 revision 的验证模型和音频，校验摘要后导出
bash scripts/prepare_test_model.sh
VAD_REFERENCE_MODEL=model-bin/source poetry run pytest -q
NCNN_TEST_ALL_CHUNKS=1 VAD_REFERENCE_MODEL=model-bin/source \
  poetry run pytest -q tests/test_native_engine.py
poetry run python scripts/benchmark_native.py --output benchmark-results/native-001
```

没有模型时相关测试跳过。`NCNN_TEST_MODEL`、`VAD_REFERENCE_MODEL`、`VAD_TEST_WAV`、
`NCNN_NUMERICAL_PROBE` 可指定资源。`tests/reference` 仅包含当前概率、特征和事件对齐测试使用的参考实现，不包含服务或调度器。
概率、缓存和 Fbank/CMVN 采用 `atol=1e-4, rtol=1e-4`，事件与水位要求一致。
测试覆盖短块继续推理、全部 1–19 帧导出配置、重采样、真实音频、状态隔离、
取消、超时、容量、结果所有权和服务退出。

非 16 kHz 离线输入由原 SciPy FFT 改为 libsoxr HQ，因此这一部分允许行为变化。
流式仍为 soxr HQ，16 kHz 路径保持原有数值与事件语义。最终 EOF 时长由原始样本数决定。

原生 CPU 基准比较默认配置和四 worker 配置，使用相同模型、输入及固定核心。
按 1/4/8/16/32 路预热并测三轮；短测低于每路 2× 实时或出现失败则停止更高并发。
原始结果保留失败记录，共享主机结果不能替代独占硬件验收。
针对已运行服务（包括容器）可使用以下命令；服务须事先绑定与客户端不重叠的核心：

```bash
poetry run python scripts/benchmark_service.py --address 127.0.0.1:50022 \
  --server-pid HOST_PID --client-cpus 4,5,6,7 --output reports/service-new-run
```

`HOST_PID` 是宿主机可见的服务进程 PID；容器可用
`docker inspect --format '{{.State.Pid}}' CONTAINER` 查询。
工具记录服务进程及子进程合计资源、吞吐、完成耗时分位数、实时节奏发送延迟和原始数据。
单进程与旧版对比必须使用同一镜像底座、模型、计算线程预算及绑核条件。
CPU 测试使用上述脚本；已删除依赖外部 Torch 服务工程的旧 `benchmark` 子命令。

历史 Python 引擎报告保留于 [两块窗口报告](reports/cpu-20261011-window2/README.md)
和 [八块窗口报告](reports/cpu-20261011/README.md)，不能视为原生实现的性能结果。
本次结果见 [原生迁移验收与基准](reports/native-20261013/README.md)。
代码来源与依赖许可见 [NOTICE.md](NOTICE.md)。

## ARM64 内置模型容器

`docker/cpu-aarch64.dockerfile` 在 ARM64 容器内编译原生库，按 `poetry.lock`
安装运行依赖。默认以 CPU FP32、4 worker × 1 线程启动 gRPC 服务。
构建前准备好 `model-bin/stream-vad` 的四个部署文件；镜像内置它们，
不需要模型挂载，也不包含源权重或转换中间文件。

在仓库根目录构建本地镜像（不会推送）：

```bash
BASE_IMAGE=registry.cn-hangzhou.aliyuncs.com/migo-dl/python:3.10.18-poetry-0-4-1-aarch64
IMAGE=fireredvad-ncnn:local-singleprocess-aarch64
docker build --platform linux/arm64 --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f docker/cpu-aarch64.dockerfile -t "$IMAGE" .
docker image inspect "$IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

仅安装了 Docker 传统构建器时，在构建命令前设置 `DOCKER_BUILDKIT=0`。
若构建依赖主机的本地代理，可为构建增加 `--network host`，并通过
`--build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY`
传入已有代理环境；这些选项只用于构建，不用于服务容器。

确认容器名和宿主机端口未占用后启动：

```bash
IMAGE=fireredvad-ncnn:local-singleprocess-aarch64
NAME=fireredvad-ncnn-arm64
HOST_PORT=50053
docker run --pull never -d --restart unless-stopped --platform linux/arm64 \
  --name "$NAME" -p "${HOST_PORT}:50051" "$IMAGE"
docker logs --tail 100 "$NAME"
docker inspect "$NAME" --format '{{json .State.Health}}'
docker exec "$NAME" poetry run python /app/docker/healthcheck.py
# 按需停止；仅操作本次创建的容器。
docker stop "$NAME"
```

`EXPOSE 50051` 为容器端口，示例宿主机端口为 `50053`。
内置模型路径是 `/app/model-bin/stream-vad`，不要用空目录覆盖它。
可通过 `-e MODEL_PATH=...`、`GRPC_HOST`、`GRPC_PORT`、`NCNN_NUM_THREADS`、
`ENGINE_WORKERS`、`NCNN_THREADS_PER_WORKER` 覆盖配置，线程乘积不得超过预算。
更改 `GRPC_PORT` 时也要相应更改端口映射；健康检查自动读取该变量。
也可以在镜像名后提供完整 CLI 命令以覆盖默认 CMD。

健康检查返回 `SERVING` 只表示就绪；业务验收应使用真实音频分别调用
`Detect` 和 `StreamingDetect`，核对片段、流结束及结果与 native API 一致，
再验证 `docker stop` 能正常回收引擎进程。模型格式、协议与测试依据为本仓库
README、`src/fireredvad_ncnn/proto/ux_vad.proto` 和服务集成测试。
