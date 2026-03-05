#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  ./run_vlm_trajectory_pipeline.sh [options]

Runs:
  1) VLM trajectory generation (agents/api_consumer.py)
  2) UV->3D lifting (lift_uv_traj_to_3d.py)
  3) Top-down visualization of generated cameras (visualize_generated_trajectory_topdown.py)
  4) Rendering from lifted cameras (render_scene_from_vlm_cameras.py)

Outputs are saved to:
  demo_tmp/agentic/<vlm_model_slug>/<run_id>/
  where run_id = YYYYMMDD_HHMMSS__<scene_slug>__<demand_slug>

Options:
  --vlm-model NAME              VLM model identifier used for generation and folder naming (required)
  --user-demand TEXT            User demand prompt passed to the VLM and saved as user_demand.txt (required)
  --data-dir PATH               Scene dir used by renderer/lift source (default: demo_tmp/davis/tennis/normalized_nofilter)
  --trajectory-json PATH        Input trajectory JSON when using --skip-generate (default: trajectory.json)
  --semantic-metadata PATH      semantic_topdown_metadata.json (default: <data-dir>/top-down-semantic-ortho-all-scene/semantic_topdown_metadata.json)
  --source-traj PATH            Source TUM trajectory for orientation/depth (default: <data-dir>/pred_traj.txt)
  --depth-mode MODE             source|mean|constant (default: source)
  --constant-depth FLOAT        Used only when --depth-mode constant (default: 0.0)
  --fps INT                     Render/video FPS (default: 25)
  --num-frames INT              Number of trajectory frames requested from VLM prompt template (default: 70)
  --skip-generate               Skip step 1 and reuse existing --trajectory-json
  -h, --help                    Show this help
EOF
}

slugify() {
  local input="$1"
  local slug
  slug="$(printf '%s' "$input" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [[ -z "$slug" ]]; then
    slug="na"
  fi
  printf '%s' "$slug"
}

DATA_DIR="demo_tmp/davis/tennis/normalized_nofilter"
TRAJECTORY_JSON="trajectory.json"
SEMANTIC_METADATA_JSON=""
SOURCE_TRAJ_TXT=""
DEPTH_MODE="source"
CONSTANT_DEPTH="0.0"
FPS="25"
NUM_FRAMES="70"
SKIP_GENERATE="0"
VLM_MODEL=""
USER_DEMAND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vlm-model)
      VLM_MODEL="$2"
      shift 2
      ;;
    --user-demand)
      USER_DEMAND="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --trajectory-json)
      TRAJECTORY_JSON="$2"
      shift 2
      ;;
    --semantic-metadata)
      SEMANTIC_METADATA_JSON="$2"
      shift 2
      ;;
    --source-traj)
      SOURCE_TRAJ_TXT="$2"
      shift 2
      ;;
    --depth-mode)
      DEPTH_MODE="$2"
      shift 2
      ;;
    --constant-depth)
      CONSTANT_DEPTH="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --skip-generate)
      SKIP_GENERATE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${VLM_MODEL// }" ]]; then
  echo "Missing required --vlm-model" >&2
  exit 1
fi
if [[ -z "${USER_DEMAND// }" ]]; then
  echo "Missing required --user-demand" >&2
  exit 1
fi
if [[ "$DEPTH_MODE" != "source" && "$DEPTH_MODE" != "mean" && "$DEPTH_MODE" != "constant" ]]; then
  echo "Invalid --depth-mode: $DEPTH_MODE (must be source|mean|constant)" >&2
  exit 1
fi
if ! [[ "$NUM_FRAMES" =~ ^[0-9]+$ ]] || [[ "$NUM_FRAMES" -le 0 ]]; then
  echo "Invalid --num-frames: $NUM_FRAMES (must be positive integer)" >&2
  exit 1
fi

if [[ -z "$SEMANTIC_METADATA_JSON" ]]; then
  SEMANTIC_METADATA_JSON="${DATA_DIR}/top-down-semantic-ortho-all-scene/semantic_topdown_metadata.json"
fi
if [[ -z "$SOURCE_TRAJ_TXT" ]]; then
  SOURCE_TRAJ_TXT="${DATA_DIR}/pred_traj.txt"
fi

if [[ ! -f "$SEMANTIC_METADATA_JSON" ]]; then
  echo "Missing semantic metadata json: $SEMANTIC_METADATA_JSON" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_TRAJ_TXT" ]]; then
  echo "Missing source trajectory txt: $SOURCE_TRAJ_TXT" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/pred_intrinsics.txt" ]]; then
  echo "Missing intrinsics for trajectory visualization: ${DATA_DIR}/pred_intrinsics.txt" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/all_points" ]]; then
  echo "Missing scene PLY directory for trajectory visualization: ${DATA_DIR}/all_points" >&2
  exit 1
fi

MODEL_SLUG="$(slugify "$VLM_MODEL")"
DEMAND_SLUG="$(slugify "$USER_DEMAND")"
DEMAND_SLUG="${DEMAND_SLUG:0:80}"

DATA_DIR_TRIMMED="${DATA_DIR%/}"
SCENE_NAME="$(basename "$(dirname "$DATA_DIR_TRIMMED")")"
DATASET_NAME="$(basename "$(dirname "$(dirname "$DATA_DIR_TRIMMED")")")"
if [[ -z "$SCENE_NAME" || "$SCENE_NAME" == "." || "$SCENE_NAME" == "/" ]]; then
  SCENE_NAME="$(basename "$DATA_DIR_TRIMMED")"
fi
if [[ -z "$DATASET_NAME" || "$DATASET_NAME" == "." || "$DATASET_NAME" == "/" ]]; then
  DATASET_NAME="scene"
fi
SCENE_SLUG="$(slugify "${DATASET_NAME}_${SCENE_NAME}")"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID_BASE="${TIMESTAMP}__${SCENE_SLUG}__${DEMAND_SLUG}"
RUN_ID="$RUN_ID_BASE"
MODEL_DIR="demo_tmp/agentic/${MODEL_SLUG}"
COUNTER=1
while [[ -e "${MODEL_DIR}/${RUN_ID}" ]]; do
  RUN_ID="${RUN_ID_BASE}_${COUNTER}"
  COUNTER=$((COUNTER + 1))
done

EXPERIMENT_DIR="${MODEL_DIR}/${RUN_ID}"
VLM_DIR="${EXPERIMENT_DIR}/01_vlm"
LIFT_DIR="${EXPERIMENT_DIR}/02_lift"
TOPDOWN_VIZ_OUTPUT_DIR="${EXPERIMENT_DIR}/03_topdown_generated_trajectory_viz"
RENDER_OUTPUT_DIR="${EXPERIMENT_DIR}/04_render_vlm_cameras"

mkdir -p "$VLM_DIR" "$LIFT_DIR" "$TOPDOWN_VIZ_OUTPUT_DIR" "$RENDER_OUTPUT_DIR"

VLM_TRAJECTORY_JSON="${VLM_DIR}/trajectory.json"
VLM_REASONING_TXT="${VLM_DIR}/reasoning.txt"
VLM_USER_DEMAND_TXT="${VLM_DIR}/user_demand.txt"
LIFTED_JSON_CANON="${LIFT_DIR}/lifted_camera_trajectory_3d.json"
LIFTED_TUM_CANON="${LIFT_DIR}/lifted_pred_traj.txt"

if [[ "$SKIP_GENERATE" == "0" ]]; then
  echo "[1/4] Generating 2D trajectory with VLM..."
  python agents/cinematographer.py \
    --model "$VLM_MODEL" \
    --user-demand "$USER_DEMAND" \
    --num-frames "$NUM_FRAMES" \
    --trajectory-out "$VLM_TRAJECTORY_JSON" \
    --reasoning-out "$VLM_REASONING_TXT" \
    --user-demand-out "$VLM_USER_DEMAND_TXT"
else
  echo "[1/4] Skipping generation step (--skip-generate)."
  if [[ ! -f "$TRAJECTORY_JSON" ]]; then
    echo "Missing trajectory json for --skip-generate: $TRAJECTORY_JSON" >&2
    exit 1
  fi
  cp "$TRAJECTORY_JSON" "$VLM_TRAJECTORY_JSON"
  printf '%s\n' "N/A (generation skipped)" > "$VLM_REASONING_TXT"
  printf '%s\n' "$USER_DEMAND" > "$VLM_USER_DEMAND_TXT"
fi

if [[ ! -f "$VLM_TRAJECTORY_JSON" ]]; then
  echo "Missing generated trajectory json: $VLM_TRAJECTORY_JSON" >&2
  exit 1
fi
if [[ ! -f "$VLM_REASONING_TXT" ]]; then
  echo "Missing reasoning text: $VLM_REASONING_TXT" >&2
  exit 1
fi
if [[ ! -f "$VLM_USER_DEMAND_TXT" ]]; then
  echo "Missing user demand text: $VLM_USER_DEMAND_TXT" >&2
  exit 1
fi

echo "[2/4] Lifting UV trajectory to 3D..."
python lift_uv_traj_to_3d.py \
  --trajectory_json "$VLM_TRAJECTORY_JSON" \
  --semantic_metadata_json "$SEMANTIC_METADATA_JSON" \
  --source_traj_txt "$SOURCE_TRAJ_TXT" \
  --output_json "$LIFTED_JSON_CANON" \
  --output_tum "$LIFTED_TUM_CANON" \
  --depth_mode "$DEPTH_MODE" \
  --constant_depth "$CONSTANT_DEPTH"

if [[ ! -f "$LIFTED_JSON_CANON" ]]; then
  echo "Missing lifted JSON output: $LIFTED_JSON_CANON" >&2
  exit 1
fi
if [[ ! -f "$LIFTED_TUM_CANON" ]]; then
  echo "Missing lifted TUM output: $LIFTED_TUM_CANON" >&2
  exit 1
fi

echo "[3/4] Visualizing generated cameras in top-down scene..."
python visualize_generated_trajectory_topdown.py \
  --semantic_metadata_json "$SEMANTIC_METADATA_JSON" \
  --generated_traj_txt "$LIFTED_TUM_CANON" \
  --source_traj_txt "$SOURCE_TRAJ_TXT" \
  --intrinsics_path "${DATA_DIR}/pred_intrinsics.txt" \
  --scene_ply_dir "${DATA_DIR}/all_points" \
  --output_dir "$TOPDOWN_VIZ_OUTPUT_DIR" \
  --fps "$FPS" \
  --draw_frustums

echo "[4/4] Rendering scene from lifted cameras..."
python render_scene_from_vlm_cameras.py \
  --data_dir "$DATA_DIR" \
  --traj_path "$LIFTED_TUM_CANON" \
  --output_dir "$RENDER_OUTPUT_DIR" \
  --fps "$FPS"

if [[ ! -f "${RENDER_OUTPUT_DIR}/vis-videos/render.mp4" ]]; then
  echo "Missing render video output: ${RENDER_OUTPUT_DIR}/vis-videos/render.mp4" >&2
  exit 1
fi

MANIFEST_PATH="${EXPERIMENT_DIR}/run_manifest.json"
CREATED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
export MANIFEST_PATH CREATED_AT EXPERIMENT_DIR RUN_ID DATA_DIR VLM_MODEL USER_DEMAND
export VLM_TRAJECTORY_JSON VLM_REASONING_TXT VLM_USER_DEMAND_TXT
export LIFTED_JSON_CANON LIFTED_TUM_CANON TOPDOWN_VIZ_OUTPUT_DIR RENDER_OUTPUT_DIR
export SEMANTIC_METADATA_JSON SOURCE_TRAJ_TXT FPS DEPTH_MODE CONSTANT_DEPTH SKIP_GENERATE NUM_FRAMES

python - <<'PY'
import json
import os
from pathlib import Path


def abs_path(path: str) -> str:
    return str(Path(path).resolve())


manifest = {
    "created_at": os.environ["CREATED_AT"],
    "run_id": os.environ["RUN_ID"],
    "experiment_dir": {
        "relative": os.environ["EXPERIMENT_DIR"],
        "absolute": abs_path(os.environ["EXPERIMENT_DIR"]),
    },
    "vlm_model": os.environ["VLM_MODEL"],
    "user_demand": os.environ["USER_DEMAND"],
    "data_dir": {
        "relative": os.environ["DATA_DIR"],
        "absolute": abs_path(os.environ["DATA_DIR"]),
    },
    "options": {
        "fps": int(os.environ["FPS"]),
        "depth_mode": os.environ["DEPTH_MODE"],
        "constant_depth": float(os.environ["CONSTANT_DEPTH"]),
        "skip_generate": os.environ["SKIP_GENERATE"] == "1",
        "num_frames": int(os.environ["NUM_FRAMES"]),
    },
    "inputs": {
        "semantic_metadata_json": {
            "relative": os.environ["SEMANTIC_METADATA_JSON"],
            "absolute": abs_path(os.environ["SEMANTIC_METADATA_JSON"]),
        },
        "source_traj_txt": {
            "relative": os.environ["SOURCE_TRAJ_TXT"],
            "absolute": abs_path(os.environ["SOURCE_TRAJ_TXT"]),
        },
    },
    "artifacts": {
        "trajectory_json": {
            "relative": os.environ["VLM_TRAJECTORY_JSON"],
            "absolute": abs_path(os.environ["VLM_TRAJECTORY_JSON"]),
        },
        "reasoning_txt": {
            "relative": os.environ["VLM_REASONING_TXT"],
            "absolute": abs_path(os.environ["VLM_REASONING_TXT"]),
        },
        "user_demand_txt": {
            "relative": os.environ["VLM_USER_DEMAND_TXT"],
            "absolute": abs_path(os.environ["VLM_USER_DEMAND_TXT"]),
        },
        "lifted_json": {
            "relative": os.environ["LIFTED_JSON_CANON"],
            "absolute": abs_path(os.environ["LIFTED_JSON_CANON"]),
        },
        "lifted_tum": {
            "relative": os.environ["LIFTED_TUM_CANON"],
            "absolute": abs_path(os.environ["LIFTED_TUM_CANON"]),
        },
        "topdown_generated_trajectory_viz_dir": {
            "relative": os.environ["TOPDOWN_VIZ_OUTPUT_DIR"],
            "absolute": abs_path(os.environ["TOPDOWN_VIZ_OUTPUT_DIR"]),
        },
        "render_vlm_cameras_dir": {
            "relative": os.environ["RENDER_OUTPUT_DIR"],
            "absolute": abs_path(os.environ["RENDER_OUTPUT_DIR"]),
        },
    },
}

Path(os.environ["MANIFEST_PATH"]).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

echo "Pipeline complete."
echo "Experiment root: $EXPERIMENT_DIR"
echo "2D trajectory: $VLM_TRAJECTORY_JSON"
echo "Reasoning: $VLM_REASONING_TXT"
echo "User demand: $VLM_USER_DEMAND_TXT"
echo "Lifted 3D JSON: $LIFTED_JSON_CANON"
echo "Lifted TUM: $LIFTED_TUM_CANON"
echo "Top-down generated trajectory viz: $TOPDOWN_VIZ_OUTPUT_DIR"
echo "Render output: $RENDER_OUTPUT_DIR"
echo "Manifest: $MANIFEST_PATH"
