#!/usr/bin/env bash
set -euo pipefail
# Explicit opt-in test resources; production never downloads a model.
cd "$(dirname "$0")/.."
mkdir -p model-bin/source
revision=b7f3207233ae36021bce31c5561549edc80e3325
curl --fail --location --retry 2 --output model-bin/source/config.json "https://huggingface.co/MigoXV/firered-vad-stream/resolve/$revision/config.json"
curl --fail --location --retry 2 --output model-bin/source/cmvn.ark "https://huggingface.co/MigoXV/firered-vad-stream/resolve/$revision/cmvn.ark"
curl --fail --location --retry 2 --output model-bin/source/model.safetensors "https://huggingface.co/MigoXV/firered-vad-stream/resolve/$revision/model.safetensors"
curl --fail --location --retry 2 --output model-bin/speech.wav https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/assets/hello_en.wav
printf '%s\n' \
  'b785f58f31d3c4fb1f3704e6cb1f9a165bd1ba26986d51b8e1a5539f755e7a6f  model-bin/source/config.json' \
  'c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473  model-bin/source/cmvn.ark' \
  '715b19620ce21e885dc7c3c399231111c761966394ea34f6bb29f1731ed95070  model-bin/source/model.safetensors' \
  '92d353353bf8955d8561d813ac0920adcabfbf6c67aa65369df44b4f86989e24  model-bin/speech.wav' | sha256sum --check
poetry run fireredvad-ncnn export model-bin/source model-bin/stream-vad
