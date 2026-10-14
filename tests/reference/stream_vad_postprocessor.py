"""Historical postprocessor oracle; production runs the C++ state machine."""
# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu)

import enum
from collections import deque
from dataclasses import dataclass


@dataclass
class StreamVadFrameResult:
    # 单帧后处理结果：保存这一帧的原始/平滑概率，以及这一帧是否触发语音段起止
    frame_idx: int  # 1-based
    is_speech: bool
    raw_prob: float
    smoothed_prob: float
    is_speech_start: bool = False
    is_speech_end: bool = False
    speech_start_frame: int = -1  # 1-based
    speech_end_frame: int = -1    # 1-based


@enum.unique
class VadState(enum.Enum):
    # 状态机：静音 -> 疑似语音 -> 已确认语音 -> 疑似静音
    SILENCE = 0
    POSSIBLE_SPEECH = 1
    SPEECH = 2
    POSSIBLE_SILENCE = 3


class StreamVadPostprocessor:
    def __init__(self,
                 smooth_window_size,
                 speech_threshold,
                 pad_start_frame,
                 min_speech_frame,
                 max_speech_frame,
                 min_silence_frame):
        # 这些参数决定：单帧概率如何平滑、连续多少帧才开段/闭段、单段最长能持续多久
        self.smooth_window_size = max(1, smooth_window_size)
        self.speech_threshold = speech_threshold
        self.pad_start_frame = max(self.smooth_window_size, pad_start_frame)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.reset()

    def reset(self):
        self.frame_cnt = 0
        # 保存最近 smooth_window_size 帧概率，用于滑动平均
        self.smooth_window = deque()
        self.smooth_window_sum = 0.0
        # speech_cnt 在开段前统计连续语音帧，在开段后统计当前段总长度；silence_cnt 统计连续静音帧
        self.state = VadState.SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        # hit_max_speech 表示上一段是否因过长被强制截断；last_* 记录最近一段边界
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def process_one_frame(self, raw_prob):
        # 先把单帧概率做平滑和过阈值，得到语音候选；是否真的开段/闭段由后面的状态机决定
        assert type(raw_prob) == float
        assert 0.0 <= raw_prob and raw_prob <= 1.0
        self.frame_cnt += 1

        smoothed_prob = self.smooth_prob(raw_prob)

        is_speech = self.apply_threshold(smoothed_prob)

        result = StreamVadFrameResult(
            frame_idx = self.frame_cnt,
            is_speech=is_speech,
            raw_prob=round(raw_prob, 3),
            smoothed_prob=round(smoothed_prob, 3)
        )

        result = self.state_transition(is_speech, result)

        return result

    def confirmed_frame_count(self) -> int:
        """返回不会再被未来边界修改的前缀帧数（右开区间）。"""
        if self.frame_cnt == 0:
            return 0
        if self.state in (VadState.SPEECH, VadState.POSSIBLE_SILENCE):
            # 疑似静音仍属于已开启的有效音段，结束位置不回退。
            return self.frame_cnt
        candidate_start = self.frame_cnt + 1
        if self.state == VadState.POSSIBLE_SPEECH:
            candidate_start -= self.speech_cnt
        earliest_start = max(
            1,
            candidate_start - self.pad_start_frame,
            self.last_speech_end_frame + 1,
        )
        return earliest_start - 1

    def smooth_prob(self, prob):
        # 用最近 N 帧平均概率做判定，避免短促噪声或单帧抖动直接触发语音
        if self.smooth_window_size <= 1:
            return prob
        self.smooth_window.append(prob)
        self.smooth_window_sum += prob
        if len(self.smooth_window) > self.smooth_window_size:
            left = self.smooth_window.popleft()
            self.smooth_window_sum -= left
        smoothed_prob = self.smooth_window_sum / len(self.smooth_window)
        return smoothed_prob

    def apply_threshold(self, prob):
        # 这里只判断当前帧像不像语音；单帧过阈值还不等于真正进入语音段
        return int(prob >= self.speech_threshold)

    def state_transition(self, is_speech, result):
        # 状态机把单帧 is_speech 候选变成段级起止事件
        # 如果上一段因为过长被强制截断，当前帧直接作为新段起点继续
        if self.hit_max_speech:
            result.is_speech_start = True
            result.speech_start_frame = self.frame_cnt
            self.last_speech_start_frame = result.speech_start_frame
            self.hit_max_speech = False

        if self.state == VadState.SILENCE:
            # 静音态下，单个语音帧还不算开段，只先进入“疑似语音”并开始累计
            if is_speech:
                self.state = VadState.POSSIBLE_SPEECH
                self.speech_cnt += 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0

        elif self.state == VadState.POSSIBLE_SPEECH:
            # 只有连续语音帧达到 min_speech_frame，才真正判定为开段；中途断掉就回退为静音
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech_frame:
                    self.state = VadState.SPEECH
                    result.is_speech_start = True
                    # 开段时向前回补 pad_start_frame 帧，但不能回补到上一段结束之前
                    result.speech_start_frame = max(1,
                        self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
                        self.last_speech_end_frame + 1)
                    self.last_speech_start_frame = result.speech_start_frame
                    self.silence_cnt = 0
            else:
                self.state = VadState.SILENCE
                self.silence_cnt = 1
                self.speech_cnt = 0

        elif self.state == VadState.SPEECH:
            # 已在语音段内时，单个静音帧不会立刻闭段，而是先转入“疑似静音”观察
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                # 当前语音段超过 max_speech_frame 时强制截断，避免单段无限增长
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
            else:
                self.state = VadState.POSSIBLE_SILENCE
                self.silence_cnt += 1

        elif self.state == VadState.POSSIBLE_SILENCE:
            # 只有连续静音帧达到 min_silence_frame 才闭段；如果中途又检测到语音，就回到语音态
            self.speech_cnt += 1
            if is_speech:
                self.state = VadState.SPEECH
                self.silence_cnt = 0
                # 当前语音段超过 max_speech_frame 时强制截断，避免单段无限增长
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame

            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = VadState.SILENCE
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_end_frame = result.speech_end_frame
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        return result
