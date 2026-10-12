import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


FIRERED_POSTPROCESSOR = "firered.stream_vad"
FIRERED_POSTPROCESSOR_VERSION = 1

_INTEGER_POSTPROCESSING_PARAMETERS = {
    "smooth_window_size",
    "pad_start_frame",
    "min_speech_frame",
    "max_speech_frame",
    "min_silence_frame",
}
_POSTPROCESSING_PARAMETERS = _INTEGER_POSTPROCESSING_PARAMETERS | {
    "speech_threshold"
}

@dataclass
class FireRedStreamVadConfig:
    use_gpu: bool = False
    smooth_window_size: int = 5  # 概率平滑窗口大小，单位为帧
    speech_threshold: float = 0.5  # 语音/静音判定阈值
    pad_start_frame : int = 5  # 检测到语音起点时向前补的帧数
    min_speech_frame: int = 8  # 判定为语音段所需的最少连续语音帧数
    max_speech_frame: int = 2000  # 单个语音段允许的最大帧数，约 20 秒
    min_silence_frame: int = 20  # 结束语音段所需的最少连续静音帧数
    chunk_max_frame: int = 30000  # full 模式下单次处理的最大帧数，约 300 秒
    def __post_init__(self):
        if self.speech_threshold < 0 or self.speech_threshold > 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if self.min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")


def resolve_postprocessing_config(
    default: FireRedStreamVadConfig,
    options: Mapping[str, Any] | None,
) -> FireRedStreamVadConfig:
    """将通用协议配置转换为 FireRed 引擎内部的强类型配置。"""
    if options is None:
        return default
    if not isinstance(options, Mapping):
        raise ValueError("postprocessing must be an object")
    if options.get("processor") != FIRERED_POSTPROCESSOR:
        raise ValueError(
            f"postprocessing.processor must be {FIRERED_POSTPROCESSOR!r}"
        )
    if options.get("version") != FIRERED_POSTPROCESSOR_VERSION:
        raise ValueError(
            "postprocessing.version must be "
            f"{FIRERED_POSTPROCESSOR_VERSION}"
        )

    parameters = options.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("postprocessing.parameters must be an object")
    if any(not isinstance(name, str) for name in parameters):
        raise ValueError("postprocessing parameter names must be strings")
    unknown = sorted(set(parameters) - _POSTPROCESSING_PARAMETERS)
    if unknown:
        raise ValueError(
            f"unknown postprocessing parameter: {unknown[0]}"
        )

    overrides: dict[str, int | float] = {}
    for name, value in parameters.items():
        path = f"postprocessing.parameters.{name}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        if name in _INTEGER_POSTPROCESSING_PARAMETERS:
            if int(value) != value or value <= 0:
                raise ValueError(f"{path} must be a positive integer")
            overrides[name] = int(value)
        else:
            if value < 0 or value > 1:
                raise ValueError(f"{path} must be in [0, 1]")
            overrides[name] = float(value)

    return replace(default, **overrides)
