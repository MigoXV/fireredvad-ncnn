#!/usr/bin/env bash
set -euo pipefail

# Build the unmodified official NCNN quantization tools against the same native
# NCNN static library as the runtime. All output paths must be on the data disk.
if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 /absolute/native-build /absolute/data-disk/tool-output" >&2
  exit 2
fi
NATIVE_BUILD_DIR="$(realpath "$1")"
QUANT_TOOL_DIR="$2"
NCNN_SOURCE_DIR="${NATIVE_BUILD_DIR}/_deps/ncnn-src"
NCNN_BUILD_DIR="${NATIVE_BUILD_DIR}/_deps/ncnn-build"
if [[ ! -f "${NCNN_BUILD_DIR}/src/libncnn.a" ]]; then
  echo "Build the native runtime first; the matching static NCNN library is required." >&2
  exit 1
fi
mkdir -p "${QUANT_TOOL_DIR}"
"${CXX:-g++}" -O2 -fopenmp -DUSE_LOCAL_IMREADWRITE \
  -I "${NCNN_SOURCE_DIR}/src" -I "${NCNN_BUILD_DIR}/src" \
  -I "${NCNN_SOURCE_DIR}/tools/quantize" \
  "${NCNN_SOURCE_DIR}/tools/quantize/ncnn2table.cpp" \
  "${NCNN_SOURCE_DIR}/tools/quantize/imreadwrite.cpp" \
  "${NCNN_BUILD_DIR}/src/libncnn.a" -o "${QUANT_TOOL_DIR}/ncnn2table"
"${CXX:-g++}" -O2 -fopenmp \
  -I "${NCNN_SOURCE_DIR}/src" -I "${NCNN_BUILD_DIR}/src" \
  -I "${NCNN_SOURCE_DIR}/tools" \
  "${NCNN_SOURCE_DIR}/tools/quantize/ncnn2int8.cpp" \
  "${NCNN_BUILD_DIR}/src/libncnn.a" -o "${QUANT_TOOL_DIR}/ncnn2int8"
