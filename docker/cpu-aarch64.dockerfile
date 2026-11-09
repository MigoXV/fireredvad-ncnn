ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS dependencies

WORKDIR /app
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 运行阶段仅补齐原生库、音频读取所需的系统运行库。
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction && poetry cache clear --all -n pypi

FROM dependencies AS builder

# CMake 固定在项目支持的 3.x；编译工具不复制到最终阶段。
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir cmake==3.31.6 ninja==1.11.1.4
COPY CMakeLists.txt CMakePresets.json ./
COPY src/native ./src/native
RUN cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_LIBDIR=lib -DFRV_WHEEL_LIB_DIR=/app/native-lib \
    && cmake --build build/native --parallel 8 \
    && ctest --test-dir build/native --output-on-failure \
    && cmake --install build/native --prefix /app/native-install \
    && cp -r /app/native-install/share/fireredvad/licenses /app/native-lib/licenses

FROM dependencies AS service

COPY README.md NOTICE.md ./
COPY src/fireredvad_ncnn ./src/fireredvad_ncnn
COPY --from=builder /app/native-lib ./src/fireredvad_ncnn/lib
COPY --from=builder /app/native-install/bin/fireredvad /usr/local/bin/fireredvad
COPY --from=builder /app/native-lib/libfireredvad.so /usr/local/lib/libfireredvad.so
COPY --from=builder /app/native-lib/libsoxr.so.0 /usr/local/lib/libsoxr.so.0
COPY docker/healthcheck.py ./docker/healthcheck.py

# 只打包经过转换与摘要校验的四个部署文件。
COPY model-bin/stream-vad/model.ncnn.param model-bin/stream-vad/model.ncnn.bin \
     model-bin/stream-vad/cmvn.ark model-bin/stream-vad/metadata.json ./model-bin/stream-vad/
RUN ldconfig && poetry install --only-root --no-interaction \
    && poetry run python -c "from fireredvad_ncnn.inferencers.inferencer import Inferencer; e = Inferencer('/app/model-bin/stream-vad'); e.close()"

ENV MODEL_PATH=/app/model-bin/stream-vad \
    DEVICE=cpu DTYPE=fp32 RUNNER=ncnn \
    GRPC_HOST=0.0.0.0 GRPC_PORT=50051 \
    NCNN_NUM_THREADS=4 ENGINE_WORKERS=4 NCNN_THREADS_PER_WORKER=1
EXPOSE 50051
HEALTHCHECK --interval=10s --timeout=8s --start-period=30s --retries=3 \
    CMD ["poetry", "run", "python", "/app/docker/healthcheck.py"]
CMD ["poetry", "run", "python", "-m", "fireredvad_ncnn.commands.app", "serve"]
