#!/usr/bin/env bash
set -euo pipefail

SRC_COMP_VISION="${1:-/Users/harshkhushi/Desktop/comp_vision}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO_ROOT/static/proctor_engine"

if [[ ! -d "$SRC_COMP_VISION" ]]; then
  echo "Source repo not found: $SRC_COMP_VISION" >&2
  exit 1
fi

mkdir -p "$ENGINE_DIR/pkg" "$ENGINE_DIR/models" "$ENGINE_DIR/ort" "$ENGINE_DIR/mediapipe/wasm" "$ENGINE_DIR/runtime"

cp "$SRC_COMP_VISION/web/pkg/proctor_wasm.js" "$ENGINE_DIR/pkg/proctor_wasm.js"
cp "$SRC_COMP_VISION/web/pkg/proctor_wasm_bg.wasm" "$ENGINE_DIR/pkg/proctor_wasm_bg.wasm"

cp "$SRC_COMP_VISION/web/public/models/proctor_yolo.onnx" "$ENGINE_DIR/models/proctor_yolo.onnx"
if [[ -f "$REPO_ROOT/face_landmarker.task" ]]; then
  cp "$REPO_ROOT/face_landmarker.task" "$ENGINE_DIR/models/face_landmarker.task"
fi

cp "$SRC_COMP_VISION/web/node_modules/onnxruntime-web/dist/ort.min.mjs" "$ENGINE_DIR/ort/ort.min.mjs"
cp "$SRC_COMP_VISION/web/node_modules/onnxruntime-web/dist/ort.min.js" "$ENGINE_DIR/ort/ort.min.js"
cp "$SRC_COMP_VISION/web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm" "$ENGINE_DIR/ort/ort-wasm-simd-threaded.wasm"
cp "$SRC_COMP_VISION/web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm" "$ENGINE_DIR/ort/ort-wasm-simd-threaded.jsep.wasm"
cp "$SRC_COMP_VISION/web/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs" "$ENGINE_DIR/ort/ort-wasm-simd-threaded.jsep.mjs"

cp "$SRC_COMP_VISION/web/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs" "$ENGINE_DIR/mediapipe/vision_bundle.mjs"
cp "$SRC_COMP_VISION/web/node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.js" "$ENGINE_DIR/mediapipe/wasm/vision_wasm_internal.js"
cp "$SRC_COMP_VISION/web/node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm" "$ENGINE_DIR/mediapipe/wasm/vision_wasm_internal.wasm"
cp "$SRC_COMP_VISION/web/node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js" "$ENGINE_DIR/mediapipe/wasm/vision_wasm_nosimd_internal.js"
cp "$SRC_COMP_VISION/web/node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm" "$ENGINE_DIR/mediapipe/wasm/vision_wasm_nosimd_internal.wasm"

esbuild_bin="$SRC_COMP_VISION/web/node_modules/.bin/esbuild"
if [[ -x "$esbuild_bin" && -d "$ENGINE_DIR/runtime/src" ]]; then
  pushd "$ENGINE_DIR/runtime" >/dev/null
  "$esbuild_bin" src/proctor_core.ts --format=esm --target=es2020 --outfile=proctor_core.js
  "$esbuild_bin" src/student_engine.ts --format=esm --target=es2020 --outfile=student_engine.js
  "$esbuild_bin" src/admin_verifier.ts --format=esm --target=es2020 --outfile=admin_verifier.js
  popd >/dev/null
fi

manifest_tmp="$(mktemp)"
version_tag="$(date +%F)"
generated_at="$(date -u +%FT%TZ)"

{
  printf '{\n'
  printf '  "version": "%s",\n' "$version_tag"
  printf '  "generated_at": "%s",\n' "$generated_at"
  printf '  "assets": {\n'

  first=1
  while IFS= read -r rel; do
    hash="$(shasum -a 256 "$ENGINE_DIR/$rel" | awk '{print $1}')"
    if [[ "$first" -eq 0 ]]; then
      printf ',\n'
    fi
    printf '    "%s": "sha256:%s"' "$rel" "$hash"
    first=0
  done < <(cd "$ENGINE_DIR" && find . -type f ! -name manifest.json | sed 's#^\./##' | sort)

  printf '\n  }\n'
  printf '}\n'
} > "$manifest_tmp"

mv "$manifest_tmp" "$ENGINE_DIR/manifest.json"

echo "Proctor assets synced to: $ENGINE_DIR"
echo "Manifest generated: $ENGINE_DIR/manifest.json"
