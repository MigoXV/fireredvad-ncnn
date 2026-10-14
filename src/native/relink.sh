#!/usr/bin/env bash
set -euo pipefail
# Usage: CXX=aarch64-linux-gnu-g++ bash relink.sh PREFIX NEW_SOXR_ARCHIVE OUTPUT [shared|cli]
prefix=${1:?installation prefix required}
soxr_archive=${2:?replacement libsoxr.a required}
output=${3:?output file required}
mode=${4:-shared}
relink_dir="$prefix/share/fireredvad/relink"
libdir="$prefix/lib"
if [[ ! -f "$libdir/libncnn.a" ]]; then libdir="$prefix/lib64"; fi
[[ -f "$libdir/libncnn.a" && -f "$soxr_archive" ]]
args=()
if [[ "$mode" == shared ]]; then
  args=(-shared -Wl,-soname,libfireredvad.so -Wl,--version-script="$relink_dir/exports.map")
elif [[ "$mode" == cli ]]; then
  args=(-I "$prefix/include" "$relink_dir/cli.cpp")
else
  printf 'unknown relink mode: %s\n' "$mode" >&2
  exit 2
fi
"${CXX:-c++}" "${args[@]}" -fPIC "$relink_dir/engine.cpp.o" \
  -Wl,--start-group "$libdir/libncnn.a" "$libdir/libfrv_fbank.a" \
  "$libdir/libfrv_kissfft.a" "$soxr_archive" -Wl,--end-group \
  -fopenmp -lm -o "$output"
