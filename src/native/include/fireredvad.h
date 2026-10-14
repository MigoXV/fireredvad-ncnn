#ifndef FIREREDVAD_H
#define FIREREDVAD_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define FRV_API __declspec(dllexport)
#else
#define FRV_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define FRV_ABI_VERSION 1u
typedef struct frv_engine frv_engine;
typedef enum frv_status {
    FRV_OK = 0,
    FRV_INVALID_ARGUMENT = 1,
    FRV_RESOURCE_EXHAUSTED = 2,
    FRV_DEADLINE_EXCEEDED = 3,
    FRV_FAILED_PRECONDITION = 4,
    FRV_UNAVAILABLE = 5,
    FRV_INTERNAL = 6,
    FRV_CANCELED = 7,
    FRV_NOT_READY = 8
} frv_status;
typedef enum frv_pcm_format { FRV_PCM_I16 = 1, FRV_PCM_F32 = 2 } frv_pcm_format;
typedef struct frv_post_config {
    uint32_t struct_size, abi_version;
    uint32_t smooth_window_size, pad_start_frame, min_speech_frame;
    uint32_t max_speech_frame, min_silence_frame, chunk_max_frame;
    double speech_threshold;
} frv_post_config;
typedef struct frv_engine_config {
    uint32_t struct_size, abi_version;
    uint32_t cpu_budget, worker_count, threads_per_worker, chunk_frames;
    uint32_t max_sessions, session_queue_capacity, request_capacity;
    uint64_t max_inflight_bytes;
} frv_engine_config;
typedef struct frv_event {
    double start, end, processed_time;
} frv_event;
typedef struct frv_result {
    uint64_t request_id;
    int32_t status;
    uint32_t reserved;
    size_t event_count;
    const frv_event *events;
    const char *message;
} frv_result;
typedef struct frv_metrics {
    uint64_t submitted, completed, rejected, canceled, expired;
    uint64_t sessions, pending, inflight_bytes, queue_high_water;
    uint64_t model_blocks, preprocess_ns, inference_ns, postprocess_ns;
} frv_metrics;
FRV_API uint32_t frv_abi_version(void);
FRV_API const char *frv_build_info(void);
/* Last error is thread-local and valid until the next ABI call on this thread. */
FRV_API const char *frv_last_error(void);
FRV_API void frv_engine_config_init(frv_engine_config *config);
FRV_API void frv_post_config_init(frv_post_config *config);
FRV_API int32_t frv_engine_create(const char *model_dir, const frv_engine_config *config,
                                  frv_engine **out);
/* Caller must stop calling the handle before destroy; destroy joins workers. */
FRV_API void frv_engine_destroy(frv_engine *engine);
FRV_API int32_t frv_session_open(frv_engine *, const frv_post_config *, int32_t endpoint_only,
                                 uint64_t *session_id);
/* Closing invalidates queued/running requests. IDs are scoped to this engine. */
FRV_API int32_t frv_session_close(frv_engine *, uint64_t session_id);
/* Input is copied before successful return. Timeout covers queue and execution;
 * timeout_ms=0 means no deadline. Sequence starts at 1, advances on acceptance.
 * session_id=0 selects offline detection. Float PCM uses int16 amplitude units.
 */
FRV_API int32_t frv_submit(frv_engine *, uint64_t session_id, uint64_t sequence, const void *pcm,
                           size_t samples, int32_t format, uint32_t sample_rate, int32_t is_final,
                           const frv_post_config *offline_config, uint32_t timeout_ms,
                           uint64_t *request_id);
/* Returns FRV_NOT_READY on wait timeout; it does not cancel the request.
 * request_id=0 consumes any completed result. A result can be consumed once.
 * Results remain charged against capacity until consumed and freed.
 */
FRV_API int32_t frv_poll(frv_engine *, uint64_t request_id, uint32_t wait_ms, frv_result **out);
FRV_API int32_t frv_cancel(frv_engine *, uint64_t request_id);
FRV_API void frv_result_free(frv_result *result);
FRV_API int32_t frv_get_metrics(frv_engine *, frv_metrics *out);
/* Sync convenience uses submit/poll, including native scheduling. */
FRV_API int32_t frv_detect(frv_engine *, const void *pcm, size_t samples, int32_t format,
                           uint32_t rate, const frv_post_config *, frv_result **out);
FRV_API int32_t frv_stream_push(frv_engine *, uint64_t session_id, uint64_t sequence,
                                const void *pcm, size_t samples, int32_t format, uint32_t rate,
                                int32_t is_final, frv_result **out);
#ifdef __cplusplus
}
#endif
#endif
