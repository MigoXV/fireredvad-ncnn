#include "fireredvad.h"
#include <algorithm>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void signal_handler(int) {
    interrupted = 1;
}
void check(int code) {
    if (code)
        throw std::runtime_error(frv_last_error());
}
uint32_t u32(const unsigned char *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint16_t u16(const unsigned char *p) {
    return uint16_t(p[0]) | uint16_t(p[1]) << 8;
}
uint32_t number(const std::string &s) {
    size_t end;
    auto n = std::stoull(s, &end);
    if (end != s.size() || n > std::numeric_limits<uint32_t>::max() || s.empty() || s[0] == '-')
        throw std::runtime_error("invalid integer: " + s);
    return static_cast<uint32_t>(n);
}
struct Audio {
    std::vector<int16_t> pcm;
    uint32_t rate;
};
Audio wav(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    unsigned char header[12];
    f.read(reinterpret_cast<char *>(header), 12);
    if (!f || std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4))
        throw std::runtime_error("expected PCM16 RIFF WAV");
    Audio audio{{}, 0};
    bool data_found = false;
    while (f) {
        unsigned char chunk[8];
        f.read(reinterpret_cast<char *>(chunk), 8);
        if (!f)
            break;
        auto n = u32(chunk + 4);
        if (!std::memcmp(chunk, "fmt ", 4)) {
            if (n < 16 || n > 4096)
                throw std::runtime_error("invalid WAV format chunk");
            std::vector<unsigned char> fmt(n);
            f.read(reinterpret_cast<char *>(fmt.data()), n);
            if (!f || u16(fmt.data()) != 1 || u16(fmt.data() + 2) != 1 ||
                u16(fmt.data() + 14) != 16 || u16(fmt.data() + 12) != 2)
                throw std::runtime_error("only mono PCM16 WAV is supported");
            audio.rate = u32(fmt.data() + 4);
        } else if (!std::memcmp(chunk, "data", 4)) {
            if (n % 2 || n > 256 * 1024 * 1024)
                throw std::runtime_error("WAV data exceeds limit or is misaligned");
            std::vector<unsigned char> data(n);
            f.read(reinterpret_cast<char *>(data.data()), n);
            if (!f)
                throw std::runtime_error("truncated WAV");
            audio.pcm.resize(n / 2);
            for (size_t i = 0; i < audio.pcm.size(); ++i)
                audio.pcm[i] = static_cast<int16_t>(u16(data.data() + 2 * i));
            data_found = true;
        } else
            f.seekg(n, std::ios::cur);
        if (n % 2)
            f.seekg(1, std::ios::cur);
    }
    if (!audio.rate || !data_found)
        throw std::runtime_error("missing WAV format or data");
    return audio;
}
void print(frv_result *raw, bool stream) {
    std::unique_ptr<frv_result, decltype(&frv_result_free)> result(raw, frv_result_free);
    if (result->status)
        throw std::runtime_error(result->message);
    std::cout << std::setprecision(17) << "{\"" << (stream ? "events" : "segments") << "\":[";
    for (size_t i = 0; i < result->event_count; ++i) {
        if (i)
            std::cout << ',';
        auto e = result->events[i];
        std::cout << '[' << e.start << ',' << e.end;
        if (stream)
            std::cout << ',' << e.processed_time;
        std::cout << ']';
    }
    std::cout << "]}\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 2 && !std::strcmp(argv[1], "--version")) {
            std::cout << frv_build_info() << '\n';
            return 0;
        }
        if (argc < 4 || !std::strcmp(argv[1], "--help")) {
            std::cerr << "Usage: fireredvad infer|stream MODEL WAV [--cpu-budget 4 --workers 1 "
                         "--threads 4 --chunk-frames 10]\n"
                         "       fireredvad stream MODEL - --raw-rate RATE [--chunk-ms 100 "
                         "--endpoint-only]\n"
                         "       --smooth-window-size N --speech-threshold P --pad-start-frame N\n"
                         "       --min-speech-frame N --max-speech-frame N --min-silence-frame N "
                         "--chunk-max-frame N\n";
            return argc == 2 ? 0 : 2;
        }
        std::string command = argv[1];
        if (command != "infer" && command != "stream")
            throw std::runtime_error("unknown command");
        bool stream = command == "stream", endpoint = false;
        uint32_t raw_rate = 0, chunk_ms = 100;
        frv_engine_config config;
        frv_engine_config_init(&config);
        frv_post_config post;
        frv_post_config_init(&post);
        for (int i = 4; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--endpoint-only") {
                endpoint = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("missing option value");
            std::string value = argv[++i];
            if (key == "--speech-threshold") {
                size_t end;
                post.speech_threshold = std::stod(value, &end);
                if (end != value.size())
                    throw std::runtime_error("invalid threshold");
                continue;
            }
            uint32_t n = number(value);
            if (key == "--cpu-budget")
                config.cpu_budget = n;
            else if (key == "--workers")
                config.worker_count = n;
            else if (key == "--threads")
                config.threads_per_worker = n;
            else if (key == "--chunk-frames")
                config.chunk_frames = n;
            else if (key == "--raw-rate")
                raw_rate = n;
            else if (key == "--chunk-ms")
                chunk_ms = n;
            else if (key == "--smooth-window-size")
                post.smooth_window_size = n;
            else if (key == "--speech-threshold")
                post.speech_threshold = n;
            else if (key == "--pad-start-frame")
                post.pad_start_frame = n;
            else if (key == "--min-speech-frame")
                post.min_speech_frame = n;
            else if (key == "--max-speech-frame")
                post.max_speech_frame = n;
            else if (key == "--min-silence-frame")
                post.min_silence_frame = n;
            else if (key == "--chunk-max-frame")
                post.chunk_max_frame = n;
            else
                throw std::runtime_error("unknown option: " + key);
        }
        frv_engine *ptr = nullptr;
        check(frv_engine_create(argv[2], &config, &ptr));
        std::unique_ptr<frv_engine, decltype(&frv_engine_destroy)> engine(ptr, frv_engine_destroy);
        struct sigaction action{};
        action.sa_handler = signal_handler;
        sigemptyset(&action.sa_mask);
        // Interrupt a blocked raw-PCM read so SIGTERM can release engine resources.
        sigaction(SIGINT, &action, nullptr);
        sigaction(SIGTERM, &action, nullptr);
        if (!stream) {
            if (raw_rate)
                throw std::runtime_error("raw input is supported by stream only");
            auto audio = wav(argv[3]);
            frv_result *result = nullptr;
            check(frv_detect(ptr, audio.pcm.data(), audio.pcm.size(), FRV_PCM_I16, audio.rate,
                             &post, &result));
            print(result, false);
        } else {
            uint64_t session;
            check(frv_session_open(ptr, &post, endpoint, &session));
            uint64_t sequence = 1;
            auto push = [&](const int16_t *p, size_t n, uint32_t rate, bool final) {
                frv_result *result = nullptr;
                check(frv_stream_push(ptr, session, sequence++, p, n, FRV_PCM_I16, rate, final,
                                      &result));
                print(result, true);
            };
            if (!chunk_ms || chunk_ms > 60000)
                throw std::runtime_error("chunk-ms must be 1..60000");
            if (raw_rate) {
                size_t count = uint64_t(raw_rate) * chunk_ms / 1000;
                if (!count || count > 16 * 1024 * 1024)
                    throw std::runtime_error("raw chunk size out of range");
                std::ifstream file;
                std::istream *input = &std::cin;
                if (std::strcmp(argv[3], "-")) {
                    file.open(argv[3], std::ios::binary);
                    if (!file)
                        throw std::runtime_error("cannot open raw input");
                    input = &file;
                }
                std::vector<int16_t> buffer(count);
                while (!interrupted &&
                       input->read(reinterpret_cast<char *>(buffer.data()), count * 2).gcount()) {
                    auto bytes = input->gcount();
                    if (bytes % 2)
                        throw std::runtime_error("odd raw PCM byte count");
                    push(buffer.data(), bytes / 2, raw_rate, false);
                }
                if (!interrupted)
                    push(nullptr, 0, raw_rate, true);
            } else {
                auto audio = wav(argv[3]);
                size_t count = uint64_t(audio.rate) * chunk_ms / 1000;
                if (!count)
                    throw std::runtime_error("chunk size is zero");
                for (size_t offset = 0; offset < audio.pcm.size() && !interrupted; offset += count)
                    push(audio.pcm.data() + offset, std::min(count, audio.pcm.size() - offset),
                         audio.rate, false);
                if (!interrupted)
                    push(nullptr, 0, audio.rate, true);
            }
            check(frv_session_close(ptr, session));
        }
        return interrupted ? 130 : 0;
    } catch (const std::exception &e) {
        std::cerr << "fireredvad: " << e.what() << '\n';
        return 1;
    }
}
