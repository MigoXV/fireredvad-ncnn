#include "fireredvad.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "net.h"
#include "nlohmann/json.hpp"
#include "picosha2.h"
#include "postprocessor.h"
#include "soxr.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace frv {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
// IDs must not alias another engine's handles, including after recreation.
std::atomic<uint64_t> next_handle_id{1};
struct Error : std::runtime_error {
    int code;
    Error(int c, const std::string &s) : std::runtime_error(s), code(c) {}
};
void require(bool ok, int code, const std::string &message) {
    if (!ok)
        throw Error(code, message);
}
uint64_t allocate_id() {
    auto id = next_handle_id.load(std::memory_order_relaxed);
    for (;;) {
        require(id && id < std::numeric_limits<uint64_t>::max(), FRV_RESOURCE_EXHAUSTED,
                "native handle namespace exhausted");
        if (next_handle_id.compare_exchange_weak(id, id + 1, std::memory_order_relaxed))
            return id;
    }
}
template <class T> void version(const T *c) {
    require(c && c->struct_size == sizeof(T) && c->abi_version == FRV_ABI_VERSION,
            FRV_INVALID_ARGUMENT, "unsupported config size or ABI version");
}
void validate(const frv_post_config &c) {
    version(&c);
    require(c.smooth_window_size && c.min_speech_frame && c.max_speech_frame &&
                c.min_silence_frame && c.chunk_max_frame && std::isfinite(c.speech_threshold) &&
                c.speech_threshold >= 0 && c.speech_threshold <= 1,
            FRV_INVALID_ARGUMENT, "invalid postprocessing config");
    require(c.smooth_window_size <= 100000 && c.pad_start_frame <= 1000000, FRV_INVALID_ARGUMENT,
            "postprocessing window too large");
}
uint64_t elapsed(Time start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
struct Accounting {
    std::atomic<uint64_t> bytes{0}, requests{0};
};
struct Lease {
    std::shared_ptr<Accounting> account;
    uint64_t bytes;
    bool request;
    Lease(std::shared_ptr<Accounting> a, uint64_t b, bool r)
        : account(std::move(a)), bytes(b), request(r) {
        account->bytes += bytes;
        if (request)
            ++account->requests;
    }
    ~Lease() {
        account->bytes -= bytes;
        if (request)
            --account->requests;
    }
};
std::array<double, 80> read_cmvn(std::istream &in, std::array<double, 80> &inv) {
    std::vector<double> stats;
    char a = 0, b = 0;
    in.get(a);
    in.get(b);
    if (a == 0 && b == 'B') {
        std::string token;
        in >> token;
        in.get();
        require(token == "DM" || token == "FM", FRV_INVALID_ARGUMENT,
                "unsupported CMVN matrix encoding");
        auto dim = [&]() {
            char size;
            int32_t value;
            in.get(size);
            in.read(reinterpret_cast<char *>(&value), 4);
            require(in.good() && size == 4, FRV_INVALID_ARGUMENT, "invalid CMVN dimension");
            return value;
        };
        require(dim() == 2 && dim() == 81, FRV_INVALID_ARGUMENT, "CMVN must be [2,81]");
        for (int i = 0; i < 162; ++i) {
            double x;
            if (token == "DM")
                in.read(reinterpret_cast<char *>(&x), 8);
            else {
                float f;
                in.read(reinterpret_cast<char *>(&f), 4);
                x = f;
            }
            require(in.good(), FRV_INVALID_ARGUMENT, "truncated CMVN matrix");
            stats.push_back(x);
        }
    } else {
        in.clear();
        in.seekg(0);
        std::string text((std::istreambuf_iterator<char>(in)), {});
        for (char &c : text)
            if (c == '[' || c == ']')
                c = ' ';
        std::istringstream values(text);
        double x;
        while (values >> x)
            stats.push_back(x);
        require(stats.size() == 162, FRV_INVALID_ARGUMENT, "CMVN must contain 162 values");
    }
    require(std::all_of(stats.begin(), stats.end(), [](double x) { return std::isfinite(x); }) &&
                stats[80] >= 1,
            FRV_INVALID_ARGUMENT, "invalid CMVN statistics");
    std::array<double, 80> means;
    for (int i = 0; i < 80; ++i) {
        means[i] = stats[i] / stats[80];
        inv[i] = 1 / std::sqrt(std::max(1e-20, stats[81 + i] / stats[80] - means[i] * means[i]));
    }
    return means;
}
struct Resampler {
    uint32_t rate = 0;
    soxr_t handle = nullptr;
    ~Resampler() {
        if (handle)
            soxr_delete(handle);
    }
    std::vector<float> process(const std::vector<float> &input, uint32_t source, bool final) {
        require(!rate || rate == source, FRV_INVALID_ARGUMENT, "sample rate changed within stream");
        if (!rate) {
            rate = source;
            if (source != 16000) {
                auto io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
                auto quality = soxr_quality_spec(SOXR_HQ, 0);
                auto runtime = soxr_runtime_spec(1);
                soxr_error_t error = nullptr;
                handle = soxr_create(source, 16000, 1, &error, &io, &quality, &runtime);
                require(handle && !error, FRV_INTERNAL, error ? error : "soxr_create failed");
            }
        }
        std::vector<float> output;
        if (!handle)
            output = input;
        else {
            size_t offset = 0;
            do {
                float buffer[8192];
                size_t used = 0, made = 0;
                // A zero-size non-final input is not an EOF signal.
                if (offset == input.size())
                    break;
                auto error = soxr_process(handle, input.data() + offset, input.size() - offset,
                                          &used, buffer, 8192, &made);
                require(!error, FRV_INTERNAL, error ? error : "");
                require(used || made, FRV_INTERNAL, "soxr made no progress");
                offset += used;
                output.insert(output.end(), buffer, buffer + made);
            } while (offset < input.size());
            if (final)
                for (;;) {
                    float buffer[8192];
                    size_t made = 0;
                    auto error = soxr_process(handle, nullptr, 0, nullptr, buffer, 8192, &made);
                    require(!error, FRV_INTERNAL, error ? error : "");
                    output.insert(output.end(), buffer, buffer + made);
                    if (!made)
                        break;
                }
        }
        for (float &x : output)
            x = static_cast<float>(std::clamp(std::nearbyint(double(x)), -32768.0, 32767.0));
        return output;
    }
};
knf::FbankOptions fbank_options() {
    knf::FbankOptions o;
    o.frame_opts.samp_freq = 16000;
    o.frame_opts.frame_length_ms = 25;
    o.frame_opts.frame_shift_ms = 10;
    o.frame_opts.dither = 0;
    o.frame_opts.snip_edges = true;
    o.mel_opts.num_bins = 80;
    o.mel_opts.debug_mel = false;
    return o;
}
struct Session {
    std::shared_ptr<Lease> lease;
    frv_post_config config;
    Postprocessor post;
    Publisher publisher;
    Resampler resampler;
    knf::OnlineFbank fbank{fbank_options()};
    int32_t read_frames = 0;
    uint64_t input_samples = 0, next_sequence = 1;
    double processed = 0;
    bool invalid = false, final_submitted = false;
    std::deque<uint64_t> queue;
    std::vector<float> remainder;
    std::array<std::vector<float>, 8> caches;
    Session(frv_post_config c, bool endpoint) : config(c), post(c) {
        publisher.endpoint_only = endpoint;
        reset_caches();
    }
    void reset_caches() {
        for (auto &c : caches)
            c.assign(128 * 19, 0);
    }
};
struct Task {
    uint64_t id = 0, session_id = 0;
    std::shared_ptr<Lease> lease;
    std::shared_ptr<Session> session;
    Time deadline = Time::max();
    uint32_t rate = 16000;
    bool final = false, initialized = false, running = false, done = false, canceled = false;
    int status = FRV_OK;
    std::string message;
    std::vector<float> audio, features;
    size_t offset = 0;
    int64_t offline_start = -1;
    std::vector<frv_event> events;
};
struct OwnedResult : frv_result {
    std::vector<frv_event> events;
    std::string message;
    std::shared_ptr<Lease> lease;
};
struct Engine {
    frv_engine_config config;
    std::array<double, 80> means, inv;
    std::vector<std::unique_ptr<ncnn::Net>> nets;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable changed;
    std::shared_ptr<Accounting> account = std::make_shared<Accounting>();
    std::unordered_map<uint64_t, std::shared_ptr<Session>> sessions;
    std::unordered_map<uint64_t, std::shared_ptr<Task>> tasks;
    std::deque<uint64_t> ready, completed;
    bool stopping = false;
    uint32_t initialized_workers = 0;
    std::exception_ptr startup_error;
    frv_metrics metrics{};

    Engine(const std::string &root, frv_engine_config c) : config(c) {
        version(&c);
        require(c.cpu_budget && c.worker_count && c.threads_per_worker &&
                    uint64_t(c.worker_count) * c.threads_per_worker <= c.cpu_budget &&
                    c.chunk_frames >= 1 && c.chunk_frames <= 19 && c.max_sessions &&
                    c.request_capacity && c.session_queue_capacity && c.max_inflight_bytes,
                FRV_INVALID_ARGUMENT, "invalid engine capacity or CPU budget");
        std::ifstream metadata(root + "/metadata.json");
        require(metadata.good(), FRV_INVALID_ARGUMENT, "cannot open metadata.json");
        nlohmann::json meta;
        metadata >> meta;
        require(meta.at("format_version") == 1 && meta.at("dtype") == "float32" &&
                    meta.at("cache_shape") == nlohmann::json::array({1, 128, 19}) &&
                    meta.at("cache_count") == 8 && meta.at("feature_dim") == 80 &&
                    meta.at("chunk_frames") == c.chunk_frames,
                FRV_INVALID_ARGUMENT, "model metadata or chunk_frames mismatch");
        for (const char *name : {"model.ncnn.param", "model.ncnn.bin", "cmvn.ark"}) {
            std::ifstream file(root + "/" + name, std::ios::binary);
            require(file.good(), FRV_INVALID_ARGUMENT, std::string("cannot open ") + name);
            picosha2::hash256_one_by_one hash;
            std::array<char, 65536> buffer;
            while (file) {
                file.read(buffer.data(), buffer.size());
                auto n = file.gcount();
                hash.process(buffer.data(), buffer.data() + n);
            }
            hash.finish();
            require(picosha2::get_hash_hex_string(hash) ==
                        meta.at("files").at(name).get<std::string>(),
                    FRV_INVALID_ARGUMENT, std::string("model checksum mismatch: ") + name);
        }
        std::ifstream cmvn(root + "/cmvn.ark", std::ios::binary);
        means = read_cmvn(cmvn, inv);
        nets.resize(c.worker_count);
        try {
            for (uint32_t i = 0; i < c.worker_count; ++i)
                workers.emplace_back([this, i, root] { start_worker(i, root); });
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(
                lock, [&] { return startup_error || initialized_workers == config.worker_count; });
            if (startup_error)
                std::rethrow_exception(startup_error);
        } catch (...) {
            stop();
            throw;
        }
    }
    ~Engine() {
        stop();
    }
    void start_worker(size_t index, const std::string &root) noexcept {
        try {
            // Load and warm up on the execution owner. OpenMP pools are thread-local;
            // warming up on the caller would create an extra unused compute pool.
            auto net = std::make_unique<ncnn::Net>();
            net->opt.num_threads = config.threads_per_worker;
            net->opt.use_vulkan_compute = false;
            net->opt.use_fp16_packed = false;
            net->opt.use_fp16_storage = false;
            net->opt.use_fp16_arithmetic = false;
            net->opt.use_bf16_storage = false;
            require(net->load_param((root + "/model.ncnn.param").c_str()) == 0 &&
                        net->load_model((root + "/model.ncnn.bin").c_str()) == 0,
                    FRV_INTERNAL, "NCNN model load failed");
            frv_post_config pc;
            frv_post_config_init(&pc);
            Session warmup(pc, false);
            model(*net, std::vector<float>(config.chunk_frames * 80, 0), 0, config.chunk_frames,
                  warmup);
            {
                std::lock_guard<std::mutex> lock(mutex);
                nets[index] = std::move(net);
                ++initialized_workers;
                changed.notify_all();
            }
            worker(index);
            nets[index].reset();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!startup_error)
                startup_error = std::current_exception();
            stopping = true;
            changed.notify_all();
        }
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            changed.notify_all();
        }
        for (auto &w : workers)
            if (w.joinable())
                w.join();
    }
    std::vector<float> model(ncnn::Net &net, const std::vector<float> &features, size_t begin,
                             size_t n, Session &s) {
        ncnn::Mat x(80, static_cast<int>(config.chunk_frames));
        require(!x.empty(), FRV_RESOURCE_EXHAUSTED, "NCNN feature allocation failed");
        x.fill(0.f);
        std::memcpy(x.data, features.data() + begin * 80, n * 80 * sizeof(float));
        auto ex = net.create_extractor();
        require(ex.input("in0", x) == 0, FRV_INTERNAL, "NCNN feature input failed");
        for (int i = 0; i < 8; ++i) {
            ncnn::Mat cache(19, 128);
            require(!cache.empty(), FRV_RESOURCE_EXHAUSTED, "NCNN cache allocation failed");
            std::memcpy(cache.data, s.caches[i].data(), 128 * 19 * sizeof(float));
            require(ex.input(("in" + std::to_string(i + 1)).c_str(), cache) == 0, FRV_INTERNAL,
                    "NCNN cache input failed");
        }
        ncnn::Mat p;
        require(ex.extract("out0", p) == 0 && p.total() >= n, FRV_INTERNAL,
                "NCNN probability output failed");
        std::vector<float> probs(n);
        std::memcpy(probs.data(), p.data, n * sizeof(float));
        for (int i = 0; i < 8; ++i) {
            ncnn::Mat value;
            require(ex.extract(("out" + std::to_string(i + 1)).c_str(), value) == 0, FRV_INTERNAL,
                    "NCNN cache output failed");
            if (value.elempack != 1) {
                ncnn::Mat unpacked;
                ncnn::convert_packing(value, unpacked, 1, net.opt);
                value = unpacked;
            }
            require(value.total() == 128 * 19, FRV_INTERNAL, "NCNN cache shape mismatch");
            const float *data = static_cast<const float *>(value.data);
            const size_t padding = config.chunk_frames - n;
            if (!padding) {
                std::memcpy(s.caches[i].data(), data, 128 * 19 * sizeof(float));
                continue;
            }
            for (int row = 0; row < 128; ++row) {
                std::array<float, 19> next;
                // Python baseline: concat(old[:chunk],new[:19-padding])[-19:].
                for (size_t j = 0; j < 19; ++j) {
                    size_t index = j + n;
                    next[j] = index < config.chunk_frames
                                  ? s.caches[i][row * 19 + index]
                                  : data[row * 19 + index - config.chunk_frames];
                }
                std::copy(next.begin(), next.end(), s.caches[i].begin() + row * 19);
            }
        }
        return probs;
    }
    void prepare(Task &t) {
        auto &s = *t.session;
        require(s.input_samples <= std::numeric_limits<uint64_t>::max() - t.audio.size(),
                FRV_INVALID_ARGUMENT, "audio duration overflow");
        s.input_samples += t.audio.size();
        auto audio = s.resampler.process(t.audio, t.rate, t.final || !t.session_id);
        // Consume incrementally and pop features to bound persistent Fbank memory.
        std::vector<float> features;
        features.swap(s.remainder);
        for (size_t begin = 0; begin < audio.size(); begin += 4096) {
            int count = std::min(size_t(4096), audio.size() - begin);
            require(s.read_frames < std::numeric_limits<int32_t>::max() - 100,
                    FRV_RESOURCE_EXHAUSTED, "stream frame limit reached");
            s.fbank.AcceptWaveform(16000, audio.data() + begin, count);
            collect(s, features);
        }
        if (t.final || !t.session_id) {
            s.fbank.InputFinished();
            collect(s, features);
        }
        size_t frames = features.size() / 80;
        if (t.session_id && !t.final) {
            size_t usable = frames / config.chunk_frames * config.chunk_frames;
            s.remainder.assign(features.begin() + usable * 80, features.end());
            features.resize(usable * 80);
        }
        t.features.swap(features);
        t.audio.clear();
        t.initialized = true;
    }
    void collect(Session &s, std::vector<float> &dest) {
        int available = s.fbank.NumFramesReady();
        for (int frame = s.read_frames; frame < available; ++frame) {
            const float *f = s.fbank.GetFrame(frame);
            for (int d = 0; d < 80; ++d)
                dest.push_back(static_cast<float>((double(f[d]) - means[d]) * inv[d]));
        }
        s.fbank.Pop(available - s.read_frames);
        s.read_frames = available;
    }
    bool step(Task &t, ncnn::Net &net, uint64_t &pre, uint64_t &infer, uint64_t &post,
              uint64_t &blocks) {
        if (!t.initialized) {
            auto start = Clock::now();
            prepare(t);
            pre += elapsed(start);
        }
        auto &s = *t.session;
        size_t total = t.features.size() / 80;
        if (t.offset < total) {
            size_t n = std::min(size_t(config.chunk_frames), total - t.offset);
            if (!t.session_id) {
                size_t boundary = s.config.chunk_max_frame - t.offset % s.config.chunk_max_frame;
                n = std::min(n, boundary);
                if (t.offset % s.config.chunk_max_frame == 0)
                    s.reset_caches();
            }
            auto start = Clock::now();
            auto probs = model(net, t.features, t.offset, n, s);
            infer += elapsed(start);
            ++blocks;
            start = Clock::now();
            for (float p : probs) {
                require(std::isfinite(p) && p >= 0 && p <= 1, FRV_INTERNAL,
                        "invalid model probability");
                auto r = s.post.process(p);
                if (t.session_id) {
                    if (r.start >= 0)
                        s.publisher.publish({frame_seconds(r.start), -1, s.processed}, t.events);
                    if (r.end >= 0)
                        s.publisher.publish({-1, frame_seconds(r.end), s.processed}, t.events);
                } else {
                    // Offline baseline uses if/elif for simultaneous start/end.
                    if (r.start >= 0)
                        t.offline_start = std::max(int64_t(0), r.start - 1);
                    else if (r.end >= 0 && t.offline_start >= 0) {
                        t.events.push_back({t.offline_start / 100.0, frame_seconds(r.end), 0});
                        t.offline_start = -1;
                    }
                }
            }
            t.offset += n;
            post += elapsed(start);
        }
        if (t.offset < total)
            return false;
        auto start = Clock::now();
        if (t.session_id) {
            // Integer nanosecond floor matches protobuf Duration semantics.
            double duration =
                static_cast<double>((s.input_samples / s.resampler.rate) * 1000000000ULL +
                                    (s.input_samples % s.resampler.rate) * 1000000000ULL /
                                        s.resampler.rate) /
                1e9;
            if (t.final && s.post.last_start >= 0)
                s.publisher.publish({-1, duration, s.processed}, t.events);
            double confirmed = t.final ? duration : std::min(s.post.confirmed() / 100.0, duration);
            require(confirmed >= s.processed, FRV_INTERNAL, "VAD watermark moved backwards");
            if (confirmed > s.processed || t.final) {
                s.processed = confirmed;
                s.publisher.publish({-1, -1, confirmed}, t.events);
            }
        } else if (t.offline_start >= 0 && s.post.frames) {
            t.events.push_back({t.offline_start / 100.0, (s.post.frames - 1) / 100.0, 0});
        }
        post += elapsed(start);
        return true;
    }
    void finish(const std::shared_ptr<Task> &t, int code, const std::string &message) {
        if (t->done)
            return;
        t->done = true;
        t->running = false;
        t->status = code;
        t->message = message;
        t->audio.clear();
        t->features.clear();
        if (code != FRV_OK)
            t->events.clear();
        if (t->session_id) {
            auto &s = *t->session;
            s.queue.erase(std::remove(s.queue.begin(), s.queue.end(), t->id), s.queue.end());
            if (code != FRV_OK) {
                s.invalid = true;
                sessions.erase(t->session_id);
            }
        }
        completed.push_back(t->id);
        ++metrics.completed;
        if (code == FRV_CANCELED)
            ++metrics.canceled;
        if (code == FRV_DEADLINE_EXCEEDED)
            ++metrics.expired;
        changed.notify_all();
    }
    void worker(size_t index) noexcept {
        for (;;) {
            std::shared_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (stopping)
                    return;
                size_t size = ready.size();
                while (size--) {
                    auto id = ready.front();
                    ready.pop_front();
                    auto it = tasks.find(id);
                    if (it == tasks.end() || it->second->done)
                        continue;
                    auto t = it->second;
                    if (t->canceled || t->session->invalid || Clock::now() >= t->deadline) {
                        finish(t,
                               t->canceled ? FRV_CANCELED
                                           : (t->session->invalid ? FRV_FAILED_PRECONDITION
                                                                  : FRV_DEADLINE_EXCEEDED),
                               "request canceled, session invalidated or deadline exceeded");
                        continue;
                    }
                    if (t->session_id && t->session->queue.front() != id) {
                        ready.push_back(id);
                        continue;
                    }
                    t->running = true;
                    task = t;
                    break;
                }
                if (!task) {
                    changed.wait_for(lock, std::chrono::milliseconds(5));
                    continue;
                }
            }
            bool done = false;
            int code = FRV_OK;
            std::string message;
            uint64_t pre = 0, infer = 0, post = 0, blocks = 0;
            try {
                done = step(*task, *nets[index], pre, infer, post, blocks);
            } catch (const Error &e) {
                code = e.code;
                message = e.what();
                done = true;
            } catch (const std::exception &e) {
                code = FRV_INTERNAL;
                message = e.what();
                done = true;
            } catch (...) {
                code = FRV_INTERNAL;
                message = "unknown native exception";
                done = true;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                metrics.preprocess_ns += pre;
                metrics.inference_ns += infer;
                metrics.postprocess_ns += post;
                metrics.model_blocks += blocks;
                task->running = false;
                if (stopping || task->canceled || task->session->invalid ||
                    Clock::now() >= task->deadline) {
                    code = task->canceled
                               ? FRV_CANCELED
                               : (Clock::now() >= task->deadline ? FRV_DEADLINE_EXCEEDED
                                                                 : FRV_FAILED_PRECONDITION);
                    message = "request canceled, session invalidated or deadline exceeded";
                    done = true;
                }
                if (done)
                    finish(task, code, message);
                else
                    ready.push_back(task->id);
                changed.notify_all();
            }
        }
    }
};
thread_local std::string last_error;
template <class F> int boundary(F &&f) noexcept {
    last_error.clear();
    try {
        f();
        return FRV_OK;
    } catch (const Error &e) {
        last_error = e.what();
        return e.code;
    } catch (const nlohmann::json::exception &e) {
        last_error = e.what();
        return FRV_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        last_error = "native allocation failed";
        return FRV_RESOURCE_EXHAUSTED;
    } catch (const std::exception &e) {
        last_error = e.what();
        return FRV_INTERNAL;
    } catch (...) {
        last_error = "unknown native exception";
        return FRV_INTERNAL;
    }
}
} // namespace frv
struct frv_engine {
    std::unique_ptr<frv::Engine> impl;
};
extern "C" {
uint32_t frv_abi_version(void) {
    return FRV_ABI_VERSION;
}
const char *frv_build_info(void) {
    return "fireredvad=0.1.0;abi=1;ncnn=20260526;fbank=b09e686;soxr=945b592;cpu=fp32";
}
const char *frv_last_error(void) {
    return frv::last_error.c_str();
}
void frv_engine_config_init(frv_engine_config *c) {
    if (c)
        *c = {sizeof(*c), FRV_ABI_VERSION, 4, 1, 4, 10, 128, 2, 2048, 256ULL * 1024 * 1024};
}
void frv_post_config_init(frv_post_config *c) {
    if (c)
        *c = {sizeof(*c), FRV_ABI_VERSION, 5, 5, 8, 2000, 20, 30000, 0.5};
}
int32_t frv_engine_create(const char *root, const frv_engine_config *config, frv_engine **out) {
    if (out)
        *out = nullptr;
    return frv::boundary([&] {
        frv::require(root && out, FRV_INVALID_ARGUMENT, "null create argument");
        frv::version(config);
        auto engine = std::make_unique<frv_engine>();
        engine->impl = std::make_unique<frv::Engine>(root, *config);
        *out = engine.release();
    });
}
void frv_engine_destroy(frv_engine *engine) {
    delete engine;
}
int32_t frv_session_open(frv_engine *engine, const frv_post_config *config, int32_t endpoint,
                         uint64_t *out) {
    if (out)
        *out = 0;
    return frv::boundary([&] {
        frv::require(engine && config && out, FRV_INVALID_ARGUMENT, "null session argument");
        frv::validate(*config);
        auto &e = *engine->impl;
        std::lock_guard<std::mutex> lock(e.mutex);
        frv::require(!e.stopping, FRV_UNAVAILABLE, "engine stopped");
        const uint64_t bytes = 256 * 1024 + uint64_t(config->smooth_window_size) * 16;
        if (e.sessions.size() >= e.config.max_sessions ||
            e.account->bytes + bytes > e.config.max_inflight_bytes) {
            ++e.metrics.rejected;
            throw frv::Error(FRV_RESOURCE_EXHAUSTED, "maximum session capacity reached");
        }
        auto lease = std::make_shared<frv::Lease>(e.account, bytes, false);
        auto session = std::make_shared<frv::Session>(*config, endpoint != 0);
        session->lease = std::move(lease);
        auto id = frv::allocate_id();
        e.sessions.emplace(id, std::move(session));
        *out = id;
    });
}
int32_t frv_session_close(frv_engine *engine, uint64_t id) {
    return frv::boundary([&] {
        frv::require(engine, FRV_INVALID_ARGUMENT, "null engine");
        auto &e = *engine->impl;
        std::lock_guard<std::mutex> lock(e.mutex);
        auto it = e.sessions.find(id);
        frv::require(it != e.sessions.end(), FRV_FAILED_PRECONDITION, "unknown stream session");
        it->second->invalid = true;
        e.sessions.erase(it);
        e.changed.notify_all();
    });
}
int32_t frv_submit(frv_engine *engine, uint64_t session_id, uint64_t sequence, const void *pcm,
                   size_t samples, int32_t format, uint32_t rate, int32_t final,
                   const frv_post_config *config, uint32_t timeout, uint64_t *out) {
    if (out)
        *out = 0;
    return frv::boundary([&] {
        frv::require(engine && out && (pcm || !samples) && rate > 0 &&
                         (format == FRV_PCM_I16 || format == FRV_PCM_F32),
                     FRV_INVALID_ARGUMENT, "invalid audio argument");
        auto &e = *engine->impl;
        std::lock_guard<std::mutex> lock(e.mutex);
        frv::require(!e.stopping, FRV_UNAVAILABLE, "engine stopped");
        std::shared_ptr<frv::Session> s;
        if (session_id) {
            auto it = e.sessions.find(session_id);
            frv::require(it != e.sessions.end(), FRV_FAILED_PRECONDITION, "unknown stream session");
            s = it->second;
            frv::require(!s->invalid && !s->final_submitted && sequence == s->next_sequence,
                         FRV_FAILED_PRECONDITION, "stream closed, finalized or sequence mismatch");
            if (s->queue.size() >= uint64_t(e.config.session_queue_capacity) + 1) {
                ++e.metrics.rejected;
                throw frv::Error(FRV_RESOURCE_EXHAUSTED, "session queue capacity reached");
            }
        } else {
            frv::require(config, FRV_INVALID_ARGUMENT, "offline config is required");
            frv::validate(*config);
        }
        long double frames = std::ceil(static_cast<long double>(samples) * 16000 / rate / 160) +
                             e.config.chunk_frames + 128;
        long double estimate =
            samples * static_cast<long double>(sizeof(float)) + frames * 2048 + 256 * 1024;
        if (!session_id)
            estimate += uint64_t(config->smooth_window_size) * 16;
        frv::require(estimate <= e.config.max_inflight_bytes &&
                         samples <= std::numeric_limits<int32_t>::max(),
                     FRV_RESOURCE_EXHAUSTED, "audio input exceeds memory or sample limit");
        uint64_t bytes = static_cast<uint64_t>(estimate);
        if (e.account->requests >= e.config.request_capacity ||
            e.account->bytes + bytes > e.config.max_inflight_bytes) {
            ++e.metrics.rejected;
            throw frv::Error(FRV_RESOURCE_EXHAUSTED, "native pending capacity reached");
        }
        auto t = std::make_shared<frv::Task>();
        t->lease = std::make_shared<frv::Lease>(e.account, bytes, true);
        t->audio.resize(samples);
        for (size_t i = 0; i < samples; ++i) {
            float value;
            if (format == FRV_PCM_I16) {
                int16_t v;
                std::memcpy(&v, static_cast<const char *>(pcm) + i * 2, 2);
                value = v;
            } else
                std::memcpy(&value, static_cast<const char *>(pcm) + i * 4, 4);
            frv::require(std::isfinite(value), FRV_INVALID_ARGUMENT,
                         "PCM contains NaN or infinity");
            t->audio[i] = value;
        }
        if (!s)
            s = std::make_shared<frv::Session>(*config, false);
        t->id = frv::allocate_id();
        t->session_id = session_id;
        t->session = s;
        t->rate = rate;
        t->final = final != 0;
        t->deadline =
            timeout ? frv::Clock::now() + std::chrono::milliseconds(timeout) : frv::Time::max();
        e.tasks.emplace(t->id, t);
        try {
            e.ready.push_back(t->id);
            if (session_id)
                s->queue.push_back(t->id);
        } catch (...) {
            e.ready.erase(std::remove(e.ready.begin(), e.ready.end(), t->id), e.ready.end());
            e.tasks.erase(t->id);
            throw;
        }
        if (session_id) {
            ++s->next_sequence;
            if (final)
                s->final_submitted = true;
        }
        *out = t->id;
        ++e.metrics.submitted;
        e.metrics.queue_high_water = std::max(e.metrics.queue_high_water, uint64_t(e.tasks.size()));
        e.changed.notify_all();
    });
}
int32_t frv_poll(frv_engine *engine, uint64_t id, uint32_t wait, frv_result **out) {
    if (out)
        *out = nullptr;
    return frv::boundary([&] {
        frv::require(engine && out, FRV_INVALID_ARGUMENT, "null poll argument");
        auto &e = *engine->impl;
        std::unique_lock<std::mutex> lock(e.mutex);
        frv::require(!id || e.tasks.count(id), FRV_FAILED_PRECONDITION,
                     "unknown or consumed request");
        auto ready = [&] {
            return e.stopping ||
                   (id ? (!e.tasks.count(id) || e.tasks.at(id)->done) : !e.completed.empty());
        };
        if (!ready() && wait)
            e.changed.wait_for(lock, std::chrono::milliseconds(wait), ready);
        frv::require(!e.stopping, FRV_UNAVAILABLE, "engine stopped");
        frv::require(!id || e.tasks.count(id), FRV_FAILED_PRECONDITION,
                     "request consumed by another caller");
        frv::require(ready(), FRV_NOT_READY, "result not ready");
        if (!id)
            id = e.completed.front();
        auto t = e.tasks.at(id);
        auto result = std::make_unique<frv::OwnedResult>();
        result->events = std::move(t->events);
        result->message = t->message;
        result->lease = t->lease;
        static_cast<frv_result &>(*result) = {id,
                                              t->status,
                                              0,
                                              result->events.size(),
                                              result->events.data(),
                                              result->message.c_str()};
        e.tasks.erase(id);
        e.completed.erase(std::remove(e.completed.begin(), e.completed.end(), id),
                          e.completed.end());
        e.ready.erase(std::remove(e.ready.begin(), e.ready.end(), id), e.ready.end());
        *out = result.release();
    });
}
int32_t frv_cancel(frv_engine *engine, uint64_t id) {
    return frv::boundary([&] {
        frv::require(engine, FRV_INVALID_ARGUMENT, "null engine");
        auto &e = *engine->impl;
        std::lock_guard<std::mutex> lock(e.mutex);
        auto it = e.tasks.find(id);
        frv::require(it != e.tasks.end(), FRV_FAILED_PRECONDITION, "unknown or consumed request");
        if (!it->second->done) {
            it->second->canceled = true;
            if (it->second->session_id) {
                it->second->session->invalid = true;
                e.sessions.erase(it->second->session_id);
            }
            e.changed.notify_all();
        }
    });
}
void frv_result_free(frv_result *result) {
    delete static_cast<frv::OwnedResult *>(result);
}
int32_t frv_get_metrics(frv_engine *engine, frv_metrics *out) {
    return frv::boundary([&] {
        frv::require(engine && out, FRV_INVALID_ARGUMENT, "null metrics argument");
        auto &e = *engine->impl;
        std::lock_guard<std::mutex> lock(e.mutex);
        *out = e.metrics;
        out->sessions = e.sessions.size();
        out->pending = e.account->requests;
        out->inflight_bytes = e.account->bytes;
    });
}
int32_t frv_detect(frv_engine *engine, const void *pcm, size_t n, int32_t format, uint32_t rate,
                   const frv_post_config *config, frv_result **out) {
    if (out)
        *out = nullptr;
    if (!out)
        return FRV_INVALID_ARGUMENT;
    uint64_t id;
    int status = frv_submit(engine, 0, 0, pcm, n, format, rate, 1, config, 0, &id);
    if (status)
        return status;
    do {
        status = frv_poll(engine, id, 1000, out);
    } while (status == FRV_NOT_READY);
    return status;
}
int32_t frv_stream_push(frv_engine *engine, uint64_t session, uint64_t sequence, const void *pcm,
                        size_t n, int32_t format, uint32_t rate, int32_t final, frv_result **out) {
    if (out)
        *out = nullptr;
    if (!out)
        return FRV_INVALID_ARGUMENT;
    uint64_t id;
    int status =
        frv_submit(engine, session, sequence, pcm, n, format, rate, final, nullptr, 0, &id);
    if (status)
        return status;
    do {
        status = frv_poll(engine, id, 1000, out);
    } while (status == FRV_NOT_READY);
    return status;
}
}
