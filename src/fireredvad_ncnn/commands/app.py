from __future__ import annotations

import asyncio
import logging
import signal
import json
from pathlib import Path

import grpc
import typer
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.configs.runtime_config import RuntimeConfig
from fireredvad_ncnn.configs.serving_config import ServingConfig
from fireredvad_ncnn.configs.stream_vad_config import FireRedStreamVadConfig

app = typer.Typer(add_completion=False, help="FireRedVAD 异步 gRPC 服务")


@app.command()
def serve(
    model_path: str = typer.Argument(
        ...,
        help="转换后的 NCNN 模型目录。",
        envvar="MODEL_PATH",
    ),
    device: str = typer.Option("cpu", envvar="DEVICE", help="推理设备：cpu。"),
    dtype: str = typer.Option("fp32", envvar="DTYPE", help="推理精度：fp32。"),
    runner: str = typer.Option(
        "ncnn",
        envvar="RUNNER",
        help="执行后端：ncnn。",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        envvar="BATCH_SIZE",
        min=1,
        help="Engine 动态批处理上限。",
    ),
    batch_wait_ms: float = typer.Option(
        5.0,
        "--batch-wait-ms",
        envvar="BATCH_WAIT_MS",
        min=0.0,
        help="动态组批最大等待时间（毫秒）。",
    ),
    num_threads: int = typer.Option(4, min=1, envvar="NCNN_NUM_THREADS"),
    stream_chunk_frame: int = typer.Option(
        10,
        "--stream-chunk-frame",
        "--chunk-size",
        envvar="CHUNK_SIZE",
        min=1,
        help="流式模型每次前向的帧数。",
    ),
    host: str = typer.Option("0.0.0.0", envvar="GRPC_HOST", help="gRPC 监听地址。"),
    port: int = typer.Option(50051, envvar="GRPC_PORT", min=1, help="gRPC 监听端口。"),
    max_concurrent_rpcs: int = typer.Option(
        256,
        "--max-concurrent-rpcs",
        "--workers",
        envvar=["GRPC_MAX_CONCURRENT_RPCS", "GRPC_WORKERS"],
        min=1,
        help="最大并发 RPC 数；--workers 为兼容别名。",
    ),
    max_message_mb: int = typer.Option(
        100,
        "--max-message-mb",
        envvar="GRPC_MAX_MESSAGE_MB",
        min=1,
        help="gRPC 单消息大小上限（MiB）。",
    ),
    endpoint_only: bool = typer.Option(
        False,
        "--endpoint-only",
        "-e",
        envvar="GRPC_ENDPOINT_ONLY",
        help="仅输出闭合的 VAD 片段。",
    ),
    engine_max_sessions: int = typer.Option(
        128,
        "--engine-max-sessions",
        envvar="ENGINE_MAX_SESSIONS",
        min=1,
    ),
    engine_session_queue_capacity: int = typer.Option(
        2,
        "--engine-session-queue-capacity",
        envvar="ENGINE_SESSION_QUEUE_CAPACITY",
        min=1,
    ),
    engine_ingress_capacity: int = typer.Option(
        2048,
        "--engine-ingress-capacity",
        envvar="ENGINE_INGRESS_CAPACITY",
        min=1,
        help="API pending 请求与 IPC ingress 容量。",
    ),
    engine_queue_timeout_ms: int = typer.Option(
        5000,
        "--engine-queue-timeout-ms",
        envvar="ENGINE_QUEUE_TIMEOUT_MS",
        min=1,
    ),
    engine_detect_queue_timeout_ms: int = typer.Option(
        30000,
        "--engine-detect-queue-timeout-ms",
        envvar="ENGINE_DETECT_QUEUE_TIMEOUT_MS",
        min=1,
        help="离线 Detect 进入设备执行前的排队期限（毫秒）。",
    ),
    engine_max_inflight_mb: int = typer.Option(
        256,
        "--engine-max-inflight-mb",
        envvar="ENGINE_MAX_INFLIGHT_MB",
        min=1,
    ),
    smooth_window_size: int = typer.Option(
        5,
        envvar="STREAM_VAD_SMOOTH_WINDOW_SIZE",
        min=1,
    ),
    speech_threshold: float = typer.Option(
        0.5,
        envvar="STREAM_VAD_SPEECH_THRESHOLD",
        min=0.0,
        max=1.0,
    ),
    pad_start_frame: int = typer.Option(
        5,
        envvar="STREAM_VAD_PAD_START_FRAME",
        min=0,
    ),
    min_speech_frame: int = typer.Option(
        8,
        envvar="STREAM_VAD_MIN_SPEECH_FRAME",
        min=1,
    ),
    max_speech_frame: int = typer.Option(
        2000,
        envvar="STREAM_VAD_MAX_SPEECH_FRAME",
        min=1,
    ),
    min_silence_frame: int = typer.Option(
        20,
        envvar="STREAM_VAD_MIN_SILENCE_FRAME",
        min=1,
    ),
    chunk_max_frame: int = typer.Option(
        30000,
        envvar="STREAM_VAD_CHUNK_MAX_FRAME",
        min=1,
    ),
    log_level: str = typer.Option("INFO", "--log-level", envvar="LOG_LEVEL"),
) -> None:
    """启动单 Worker 的 NCNN CPU 异步 VAD 服务。"""
    try:
        logging_level = getattr(logging, log_level.upper(), None)
        if logging_level is None:
            raise ValueError("invalid log level")
        runtime_config = RuntimeConfig(
            device=device, dtype=dtype, runner_name=runner,
            num_threads=num_threads, stream_chunk_frame=stream_chunk_frame,
        )
        serving_config = ServingConfig(
            host=host,
            port=port,
            max_concurrent_rpcs=max_concurrent_rpcs,
            max_message_mb=max_message_mb,
        )
        engine_config = EngineConfig(
            max_batch_size=batch_size,
            max_batch_wait_ms=batch_wait_ms,
            max_sessions=engine_max_sessions,
            session_queue_capacity=engine_session_queue_capacity,
            ingress_capacity=engine_ingress_capacity,
            queue_timeout_ms=engine_queue_timeout_ms,
            detect_queue_timeout_ms=engine_detect_queue_timeout_ms,
            max_inflight_mb=engine_max_inflight_mb,
        )
        stream_vad_config = FireRedStreamVadConfig(
            use_gpu=False,
            smooth_window_size=smooth_window_size,
            speech_threshold=speech_threshold,
            pad_start_frame=pad_start_frame,
            min_speech_frame=min_speech_frame,
            max_speech_frame=max_speech_frame,
            min_silence_frame=min_silence_frame,
            chunk_max_frame=chunk_max_frame,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    asyncio.run(
        _serve_async(
            model_path,
            runtime_config,
            serving_config,
            engine_config,
            stream_vad_config,
            endpoint_only,
            logging_level,
        )
    )


async def _serve_async(
    model_path: str,
    runtime_config: RuntimeConfig,
    serving_config: ServingConfig,
    engine_config: EngineConfig,
    stream_vad_config: FireRedStreamVadConfig,
    endpoint_only: bool,
    logging_level: int,
) -> None:
    import fireredvad_ncnn.proto.ux_vad_pb2_grpc as pb_grpc
    from fireredvad_ncnn.engine.client import EngineClient
    from fireredvad_ncnn.engine.process import EngineProcess
    from fireredvad_ncnn.servicer.servicer import UxVoiceActivityDetectorService

    logger = logging.getLogger(__name__)
    process = EngineProcess(
        model_path=model_path,
        runtime_config=runtime_config,
        engine_config=engine_config,
        stream_vad_config=stream_vad_config,
        endpoint_only=endpoint_only,
        max_payload_bytes=serving_config.max_message_bytes,
        log_level=logging_level,
    )
    client = EngineClient(process.endpoint, engine_config)
    server = None
    health_service = health.aio.HealthServicer()
    service_name = "ux_vad.UxVoiceActivityDetector"
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, shutdown_event.set)

    try:
        process.start()
        await client.start()
        ready = await client.wait_ready(
            engine_config.startup_timeout_s,
            process.is_alive,
        )
        options = [
            ("grpc.max_receive_message_length", serving_config.max_message_bytes),
            ("grpc.max_send_message_length", serving_config.max_message_bytes),
        ]
        server = grpc.aio.server(
            options=options,
            maximum_concurrent_rpcs=serving_config.max_concurrent_rpcs,
        )
        service = UxVoiceActivityDetectorService(
            client,
            endpoint_only=endpoint_only,
            max_message_bytes=serving_config.max_message_bytes,
        )
        pb_grpc.add_UxVoiceActivityDetectorServicer_to_server(service, server)
        health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
        bound_port = server.add_insecure_port(
            f"{serving_config.host}:{serving_config.port}"
        )
        if bound_port == 0:
            raise RuntimeError("gRPC failed to bind the configured address")
        await server.start()
        await health_service.set("", health_pb2.HealthCheckResponse.SERVING)
        await health_service.set(service_name, health_pb2.HealthCheckResponse.SERVING)
        logger.info(
            "FireRedVAD ready: address=%s:%s worker_pid=%s generation=%s device=%s dtype=%s runner=%s batch=%s wait_ms=%s",
            serving_config.host,
            bound_port,
            ready.get("pid"),
            ready.get("generation"),
            runtime_config.device,
            runtime_config.dtype,
            runtime_config.runner_name,
            engine_config.max_batch_size,
            engine_config.max_batch_wait_ms,
        )
        monitor_task = asyncio.create_task(
            _monitor_worker(process, shutdown_event),
            name="engine-monitor",
        )
        await shutdown_event.wait()
        await monitor_task
    finally:
        if server is not None:
            await health_service.enter_graceful_shutdown()
            await server.stop(engine_config.shutdown_grace_s)
        await client.shutdown_worker()
        await client.close()
        process.join_or_terminate()


async def _monitor_worker(process, shutdown_event: asyncio.Event) -> None:
    while not shutdown_event.is_set():
        if not process.is_alive():
            logging.getLogger(__name__).error(
                "Engine worker exited unexpectedly: exitcode=%s",
                process.exitcode,
            )
            shutdown_event.set()
            return
        await asyncio.sleep(0.2)


@app.command("export")
def export_command(
    source: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Argument(...),
    chunk_frames: int = typer.Option(10, min=1, max=19),
):
    """将本地 HF Stream-VAD 权重转换为 NCNN（需要 export 依赖组）。"""
    from fireredvad_ncnn.exporting.convert import export_model
    export_model(source, output, chunk_frames)
    typer.echo(str(output / "metadata.json"))


@app.command("infer")
def infer_command(
    model_path: Path = typer.Argument(..., exists=True, envvar="MODEL_PATH"),
    audio_path: Path = typer.Argument(..., exists=True),
    num_threads: int = typer.Option(4, min=1, envvar="NCNN_NUM_THREADS"),
):
    """识别单声道音频，输出语音片段（秒）。"""
    import soundfile as sf
    from fireredvad_ncnn.inferencers.inferencer import Inferencer
    audio, sr = sf.read(audio_path, dtype="int16")
    if audio.ndim != 1:
        raise typer.BadParameter("Audio must be mono")
    engine = Inferencer(model_path, ncpu=num_threads)
    try:
        typer.echo(json.dumps({"segments": engine.infer(audio, sr)}, ensure_ascii=False))
    finally:
        engine.close()


@app.command("benchmark")
def benchmark_command(
    model_path: Path = typer.Argument(..., exists=True, envvar="MODEL_PATH"),
    reference_project: Path = typer.Option(Path("/workspace/opusi/fireredvad"), exists=True),
    reference_model: Path = typer.Option(Path("/workspace/opusi/fireredvad/model-bin/MigoXV/firered-vad-stream"), exists=True),
    wav_dir: Path = typer.Option(Path("/workspace/opusi/fireredvad/data-bin/0713"), exists=True),
    output: Path = typer.Option(Path("benchmark-results")),
):
    """短时比较两个 CPU gRPC 服务的 1/4/8/16/32 路吞吐。"""
    from fireredvad_ncnn.benchmark import benchmark
    benchmark(model_path, reference_project, reference_model, wav_dir, output)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
