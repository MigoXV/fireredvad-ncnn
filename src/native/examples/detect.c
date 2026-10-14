#include "fireredvad.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    frv_engine_config config;
    frv_engine_config_init(&config);
    frv_post_config post;
    frv_post_config_init(&post);
    frv_engine *engine = NULL;
    frv_result *result = NULL;
    int16_t silence[16000] = {0};
    int status = frv_engine_create(argv[1], &config, &engine);
    if (!status)
        status = frv_detect(engine, silence, 16000, FRV_PCM_I16, 16000, &post, &result);
    if (!status && result)
        status = result->status;
    if (status)
        fprintf(stderr, "%s\n", result ? result->message : frv_last_error());
    else
        printf("segments=%zu\n", result->event_count);
    frv_result_free(result);
    frv_engine_destroy(engine);
    return status ? 1 : 0;
}
