"""PEP 517 backend: native CMake build plus Poetry packaging.

Use `poetry run python -m build`. Wheels are py3-none-linux_<arch>, never
mislabelled as pure Python or manylinux. Cross wheels require an explicit toolchain.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

from poetry.core.masonry import api as poetry_api

ROOT = Path(__file__).resolve().parent


def _native_build():
    arch = os.environ.get("FIREREDVAD_BUILD_ARCH", platform.machine())
    if arch not in ("x86_64", "aarch64") or platform.system() != "Linux":
        raise RuntimeError("Only Linux x86_64 and aarch64 native wheels are supported")
    build = Path(os.environ.get("FIREREDVAD_BUILD_DIR", ROOT / "build" / ("wheel-" + arch))).resolve()
    libdir = ROOT / "src/fireredvad_ncnn/lib"
    cmake = os.environ.get("FIREREDVAD_CMAKE", "cmake")
    args = [cmake, "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DFRV_BUILD_SHARED=ON", "-DFRV_STATIC_DEPS=ON", "-DFRV_STATIC_SOXR=OFF",
            "-DFRV_WHEEL_LIB_DIR=" + str(libdir), "-Wno-dev"]
    # Reuse downloaded sources, while keeping isolated build tools out of dev caches.
    reference = ROOT / "build/x86_64/_deps"
    for name in ("ncnn", "fbank", "soxr", "json", "sha", "kissfft"):
        source = reference / f"{name}-src"
        if source.is_dir():
            args.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={source}")
    if arch != platform.machine():
        if arch != "aarch64":
            raise RuntimeError("Cross compilation is only provided for aarch64")
        args.append("-DCMAKE_TOOLCHAIN_FILE=" + str(ROOT / "src/native/cmake/aarch64-linux-gnu.cmake"))
    subprocess.run(args, check=True)
    subprocess.run([cmake, "--build", str(build), "--target", "fireredvad_shared", "-j",
                    os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")], check=True)
    library = build / "libfireredvad.so"
    header = library.read_bytes()[:20]
    if header[:4] != b"\x7fELF" or int.from_bytes(header[18:20], "little") != (183 if arch == "aarch64" else 62):
        raise RuntimeError("CMake output architecture does not match the wheel platform")
    libdir.mkdir(parents=True, exist_ok=True)
    # A cached target does not run its POST_BUILD copy; always stage explicitly.
    shutil.copyfile(library, libdir / "libfireredvad.so")
    shutil.copyfile(build / "_deps/soxr-build/src/libsoxr.so.0", libdir / "libsoxr.so.0")
    # Include licenses alongside bundled shared objects.
    licenses = libdir / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    sources = {"ncnn": "LICENSE.txt", "fbank": "LICENSE", "kissfft": "COPYING",
               "json": "LICENSE.MIT", "sha": "LICENSE", "soxr": "LICENCE"}
    cache = (build / "CMakeCache.txt").read_text()
    for name, filename in sources.items():
        override = next((line.split("=", 1)[1] for line in cache.splitlines()
                         if line.startswith(f"FETCHCONTENT_SOURCE_DIR_{name.upper()}:")), "")
        source = Path(override) if override else build / "_deps" / f"{name}-src"
        shutil.copyfile(source / filename, licenses / f"{name}.txt")
    return "linux_" + arch


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    tag = "py3-none-" + _native_build()
    filename = poetry_api.build_wheel(wheel_directory, config_settings, metadata_directory)
    pure = Path(wheel_directory) / filename
    with zipfile.ZipFile(pure) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in entries if name.endswith(".dist-info/WHEEL"))
    entries[metadata] = entries[metadata].replace(b"Root-Is-Purelib: true", b"Root-Is-Purelib: false").replace(
        b"Tag: py3-none-any", ("Tag: " + tag).encode())
    record = next(name for name in entries if name.endswith(".dist-info/RECORD"))
    rows = []
    for name, data in entries.items():
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append((name, "sha256=" + digest, str(len(data))))
    rows.append((record, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    entries[record] = output.getvalue().encode()
    native = Path(wheel_directory) / filename.replace("py3-none-any", tag)
    with zipfile.ZipFile(native, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    pure.unlink()
    return native.name


build_sdist = poetry_api.build_sdist
get_requires_for_build_wheel = poetry_api.get_requires_for_build_wheel
get_requires_for_build_sdist = poetry_api.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = poetry_api.prepare_metadata_for_build_wheel
