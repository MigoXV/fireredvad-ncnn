// FireRedVAD state machine, ported from FireRedTeam/FireRedVAD.
// Copyright 2026 Xiaohongshu (Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu).
#pragma once
#include "fireredvad.h"
#include <algorithm>
#include <deque>
#include <vector>

namespace frv {
struct FrameResult {
    int64_t frame = 0, start = -1, end = -1;
};
class Postprocessor {
  public:
    explicit Postprocessor(frv_post_config config) : c(config) {
        c.pad_start_frame = std::max(c.pad_start_frame, c.smooth_window_size);
    }
    FrameResult process(double p) {
        ++frames;
        double smooth = p;
        if (c.smooth_window_size > 1) {
            window.push_back(p);
            sum += p;
            if (window.size() > c.smooth_window_size) {
                sum -= window.front();
                window.pop_front();
            }
            smooth = sum / window.size();
        }
        bool speech = smooth >= c.speech_threshold;
        FrameResult r{frames, -1, -1};
        if (hit_max) {
            r.start = frames;
            last_start = frames;
            hit_max = false;
        }
        switch (state) {
        case Silence:
            if (speech) {
                state = PossibleSpeech;
                ++speech_count;
            } else {
                ++silence_count;
                speech_count = 0;
            }
            break;
        case PossibleSpeech:
            if (speech) {
                ++speech_count;
                if (speech_count >= c.min_speech_frame) {
                    state = Speech;
                    r.start = std::max(
                        {int64_t(1), frames - speech_count + 1 - c.pad_start_frame, last_end + 1});
                    last_start = r.start;
                    silence_count = 0;
                }
            } else {
                state = Silence;
                silence_count = 1;
                speech_count = 0;
            }
            break;
        case Speech:
            ++speech_count;
            if (speech) {
                silence_count = 0;
                truncate(r);
            } else {
                state = PossibleSilence;
                ++silence_count;
            }
            break;
        case PossibleSilence:
            ++speech_count;
            if (speech) {
                state = Speech;
                silence_count = 0;
                truncate(r);
            } else if (++silence_count >= c.min_silence_frame) {
                state = Silence;
                r.end = frames;
                last_end = frames;
                last_start = -1;
                speech_count = 0;
            }
            break;
        }
        return r;
    }
    int64_t confirmed() const {
        if (!frames)
            return 0;
        if (state == Speech || state == PossibleSilence)
            return frames;
        int64_t candidate = frames + 1;
        if (state == PossibleSpeech)
            candidate -= speech_count;
        return std::max({int64_t(1), candidate - c.pad_start_frame, last_end + 1}) - 1;
    }
    int64_t frames = 0, last_start = -1, last_end = -1;

  private:
    void truncate(FrameResult &r) {
        if (speech_count >= c.max_speech_frame) {
            hit_max = true;
            speech_count = 0;
            r.end = frames;
            last_start = -1;
            last_end = frames;
        }
    }
    enum State { Silence, PossibleSpeech, Speech, PossibleSilence };
    frv_post_config c;
    State state = Silence;
    std::deque<double> window;
    double sum = 0;
    int64_t speech_count = 0, silence_count = 0;
    bool hit_max = false;
};
struct Publisher {
    bool endpoint_only = false;
    double pending_start = -1, processed = 0;
    void publish(frv_event e, std::vector<frv_event> &out) {
        if (!endpoint_only) {
            processed = e.processed_time;
            out.push_back(e);
            return;
        }
        if (e.start >= 0)
            pending_start = e.start;
        if (e.end >= 0 && pending_start >= 0) {
            out.push_back({pending_start, e.end, processed});
            pending_start = -1;
        }
        if (pending_start >= 0)
            e.processed_time = std::min(e.processed_time, pending_start);
        if (e.processed_time > processed || (e.start < 0 && e.end < 0)) {
            processed = e.processed_time;
            out.push_back({-1, -1, processed});
        }
    }
};
inline double frame_seconds(int64_t f) {
    return std::max(int64_t(0), f - 1) / 100.0;
}
} // namespace frv
