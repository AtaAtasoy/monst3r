#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  ./run_vlm_trajectory_pipeline.sh [options]

Runs:
  1) VLM trajectory generation (agents/cinematographer.py)
  2) UV->3D lifting (lift_uv_traj_to_3d.py)
  3) Top-down visualization of generated NPZ trajectory (visualize_custom_trajectory.py)
  4) NPZ trajectory conversion and top-down visualization (visualize_generated_trajectory_topdown.py)
  5) Top-down visualization of generated cameras (TUM) (visualize_generated_trajectory_topdown.py)
  6) Rendering from lifted cameras (render_scene_from_vlm_cameras.py)
  7) Critic evaluation (agents/critic.py)

Outputs are saved to:
  demo_tmp/agentic/<vlm_model_slug>/<run_id>/iterations/iter_xxx/
  where run_id = YYYYMMDD_HHMMSS__<scene_slug>__<demand_slug>

Options:
  --vlm-model NAME              VLM model identifier used for generation and folder naming (required)
  --critic-model NAME           Critic model identifier used for evaluation (required)
  --user-demand TEXT            User demand prompt passed to the VLM and saved as user_demand.txt (required)
  --loop-mode MODE              single|iterative (default: iterative)
  --max-iterations INT          Maximum loop iterations (default: 3)
  --score-threshold FLOAT       Stop loop when score >= threshold (default: 8.0)
  --data-dir PATH               Scene dir used by renderer/lift source (default: demo_tmp/davis/tennis/normalized_nofilter)
  --dense-trajectory-json PATH  Input dense trajectory JSON when using --skip-generate (default: camera_trajectory_pixels.json)
  --strided-trajectory-json PATH
                                Input sparse trajectory JSON when using --skip-generate
  --semantic-metadata PATH      semantic_topdown_metadata.json (default: <data-dir>/top-down-semantic-ortho-all-scene/semantic_topdown_metadata.json)
  --source-traj PATH            Source TUM trajectory for orientation/depth (default: <data-dir>/pred_traj.txt)
  --depth-mode MODE             source|mean|constant (default: source)
  --constant-depth FLOAT        Used only when --depth-mode constant (default: 0.0)
  --fps INT                     Render/video FPS (default: 25)
  --num-frames INT              Number of trajectory frames requested from VLM prompt template (default: 70)
  --sparse-frame-divisor INT    Sparse control cadence for VLM generation: 2 or 4 (default: 4)
  --skip-critic                 Skip step 5 (single mode only)
  --critic-result-json PATH     Existing critic_result.json to copy when using --skip-critic (single mode only)
  --skip-generate               Skip step 1 and reuse existing split trajectory artifacts (single mode only)
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
DENSE_TRAJECTORY_JSON="camera_trajectory_pixels.json"
STRIDED_TRAJECTORY_JSON=""
SEMANTIC_METADATA_JSON=""
SOURCE_TRAJ_TXT=""
DEPTH_MODE="source"
CONSTANT_DEPTH="0.0"
FPS="25"
NUM_FRAMES="70"
SPARSE_FRAME_DIVISOR="4"
SKIP_GENERATE="0"
SKIP_CRITIC="0"
CRITIC_RESULT_JSON=""
VLM_MODEL=""
CRITIC_MODEL=""
USER_DEMAND=""
LOOP_MODE="iterative"
MAX_ITERATIONS="3"
SCORE_THRESHOLD="8.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vlm-model)
      VLM_MODEL="$2"
      shift 2
      ;;
    --critic-model)
      CRITIC_MODEL="$2"
      shift 2
      ;;
    --user-demand)
      USER_DEMAND="$2"
      shift 2
      ;;
    --loop-mode)
      LOOP_MODE="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --score-threshold)
      SCORE_THRESHOLD="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --dense-trajectory-json|--trajectory-json)
      DENSE_TRAJECTORY_JSON="$2"
      shift 2
      ;;
    --strided-trajectory-json)
      STRIDED_TRAJECTORY_JSON="$2"
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
    --sparse-frame-divisor)
      SPARSE_FRAME_DIVISOR="$2"
      shift 2
      ;;
    --skip-critic)
      SKIP_CRITIC="1"
      shift
      ;;
    --critic-result-json)
      CRITIC_RESULT_JSON="$2"
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
if [[ -z "${CRITIC_MODEL// }" ]]; then
  echo "Missing required --critic-model" >&2
  exit 1
fi
if [[ -z "${USER_DEMAND// }" ]]; then
  echo "Missing required --user-demand" >&2
  exit 1
fi
if [[ "$LOOP_MODE" != "single" && "$LOOP_MODE" != "iterative" ]]; then
  echo "Invalid --loop-mode: $LOOP_MODE (must be single|iterative)" >&2
  exit 1
fi
if ! [[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$MAX_ITERATIONS" -lt 1 ]]; then
  echo "Invalid --max-iterations: $MAX_ITERATIONS (must be integer >= 1)" >&2
  exit 1
fi
if ! python - "$SCORE_THRESHOLD" <<'PY'
import sys
try:
    v = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if 0.0 <= v <= 10.0 else 1)
PY
then
  echo "Invalid --score-threshold: $SCORE_THRESHOLD (must be in [0, 10])" >&2
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
if [[ "$SPARSE_FRAME_DIVISOR" != "2" && "$SPARSE_FRAME_DIVISOR" != "4" ]]; then
  echo "Invalid --sparse-frame-divisor: $SPARSE_FRAME_DIVISOR (must be 2 or 4)" >&2
  exit 1
fi
if [[ "$LOOP_MODE" == "iterative" && "$SKIP_GENERATE" == "1" ]]; then
  echo "--skip-generate is not supported in iterative mode." >&2
  exit 1
fi
if [[ "$LOOP_MODE" == "iterative" && "$SKIP_CRITIC" == "1" ]]; then
  echo "--skip-critic is not supported in iterative mode." >&2
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
ITERATIONS_DIR="${EXPERIMENT_DIR}/iterations"
mkdir -p "$ITERATIONS_DIR"

EFFECTIVE_MAX_ITER="$MAX_ITERATIONS"
if [[ "$LOOP_MODE" == "single" ]]; then
  EFFECTIVE_MAX_ITER="1"
fi

STOP_REASON="max_iterations_reached"
EXECUTED_ITERATIONS=0
BEST_ITER_ID=""
FINAL_ITER_ID=""
PREV_CRITIC_JSON=""
PREV_STRIDED_TRAJECTORY_JSON=""

for ((i=0; i<EFFECTIVE_MAX_ITER; i++)); do
  ITER_ID="$(printf 'iter_%03d' "$i")"
  ITER_DIR="${ITERATIONS_DIR}/${ITER_ID}"
  VLM_DIR="${ITER_DIR}/01_vlm"
  LIFT_DIR="${ITER_DIR}/02_lift"
  NPZ_VIZ_3D_OUTPUT_DIR="${ITER_DIR}/03_lifted_npz_3d_viz"
  TOPDOWN_VIZ_OUTPUT_DIR="${ITER_DIR}/03_topdown_generated_trajectory_viz"
  TOPDOWN_NPZ_VIZ_OUTPUT_DIR="${ITER_DIR}/03_topdown_generated_npz_viz"
  RENDER_OUTPUT_DIR="${ITER_DIR}/04_render_vlm_cameras"
  CRITIC_OUTPUT_DIR="${ITER_DIR}/05_critic"
  mkdir -p "$VLM_DIR" "$LIFT_DIR" "$NPZ_VIZ_3D_OUTPUT_DIR" "$TOPDOWN_VIZ_OUTPUT_DIR" "$TOPDOWN_NPZ_VIZ_OUTPUT_DIR" "$RENDER_OUTPUT_DIR" "$CRITIC_OUTPUT_DIR"

  VLM_STRIDED_TRAJECTORY_JSON="${VLM_DIR}/strided_camera_trajectory_pixels.json"
  VLM_DENSE_TRAJECTORY_JSON="${VLM_DIR}/camera_trajectory_pixels.json"
  VLM_REASONING_TXT="${VLM_DIR}/reasoning.txt"
  VLM_USER_DEMAND_TXT="${VLM_DIR}/user_demand.txt"
  LIFTED_JSON_CANON="${LIFT_DIR}/lifted_camera_trajectory_3d.json"
  LIFTED_TUM_CANON="${LIFT_DIR}/lifted_pred_traj.txt"
  LIFTED_NPZ_CANON="${LIFT_DIR}/lifted_pred_traj.npz"
  CRITIC_RESULT_CANON="${CRITIC_OUTPUT_DIR}/critic_result.json"
  CRITIC_REASONING_CANON="${CRITIC_OUTPUT_DIR}/reasoning.txt"
  CRITIC_INPUTS_DIR="${CRITIC_OUTPUT_DIR}/inputs"

  echo "=== Iteration ${ITER_ID} ==="
  if [[ "$SKIP_GENERATE" == "0" ]]; then
  echo "[${ITER_ID}][1/7] Generating 2D trajectory with VLM..."
    GEN_CMD=(
      python -m agents.cinematographer
      --model "$VLM_MODEL"
      --user-demand "$USER_DEMAND"
      --num-frames "$NUM_FRAMES"
      --sparse-frame-divisor "$SPARSE_FRAME_DIVISOR"
      --strided-trajectory-out "$VLM_STRIDED_TRAJECTORY_JSON"
      --dense-trajectory-out "$VLM_DENSE_TRAJECTORY_JSON"
      --reasoning-out "$VLM_REASONING_TXT"
      --user-demand-out "$VLM_USER_DEMAND_TXT"
    )
    if [[ "$i" -gt 0 ]]; then
      GEN_CMD+=(--critic-feedback-json "$PREV_CRITIC_JSON")
      GEN_CMD+=(--prior-strided-trajectory-json "$PREV_STRIDED_TRAJECTORY_JSON")
    fi
    CUDA_VISIBLE_DEVICES=1 "${GEN_CMD[@]}"
  else
    echo "[${ITER_ID}][1/7] Skipping generation step (--skip-generate)."
    if [[ ! -f "$DENSE_TRAJECTORY_JSON" ]]; then
      echo "Missing dense trajectory json for --skip-generate: $DENSE_TRAJECTORY_JSON" >&2
      exit 1
    fi
    if [[ ! -f "$STRIDED_TRAJECTORY_JSON" ]]; then
      echo "Missing strided trajectory json for --skip-generate: $STRIDED_TRAJECTORY_JSON" >&2
      exit 1
    fi
    cp "$DENSE_TRAJECTORY_JSON" "$VLM_DENSE_TRAJECTORY_JSON"
    cp "$STRIDED_TRAJECTORY_JSON" "$VLM_STRIDED_TRAJECTORY_JSON"
    printf '%s\n' "N/A (generation skipped)" > "$VLM_REASONING_TXT"
    printf '%s\n' "$USER_DEMAND" > "$VLM_USER_DEMAND_TXT"
  fi

  if [[ ! -f "$VLM_STRIDED_TRAJECTORY_JSON" ]]; then
    echo "Missing generated strided trajectory json: $VLM_STRIDED_TRAJECTORY_JSON" >&2
    exit 1
  fi
  if [[ ! -f "$VLM_DENSE_TRAJECTORY_JSON" ]]; then
    echo "Missing generated dense trajectory json: $VLM_DENSE_TRAJECTORY_JSON" >&2
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

  echo "[${ITER_ID}][2/7] Lifting UV trajectory to 3D..."
  CUDA_VISIBLE_DEVICES=1 python lift_uv_traj_to_3d.py \
    --trajectory_json "$VLM_DENSE_TRAJECTORY_JSON" \
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
  echo "[${ITER_ID}][2/7] Converting lifted trajectory to NPZ..."
  CUDA_VISIBLE_DEVICES=1 python agents/lifted_trajectory_to_npz.py \
    --lifted-json "$LIFTED_JSON_CANON" \
    --output-npz "$LIFTED_NPZ_CANON"
  if [[ ! -f "$LIFTED_NPZ_CANON" ]]; then
    echo "Missing lifted NPZ output: $LIFTED_NPZ_CANON" >&2
    exit 1
  fi
  echo "[${ITER_ID}][3/7] Visualizing lifted NPZ trajectory in 3D..."
  CUDA_VISIBLE_DEVICES=1 python visualize_custom_trajectory.py \
    --pose_path "$LIFTED_NPZ_CANON" \
    --output_path "${NPZ_VIZ_3D_OUTPUT_DIR}/lifted_pred_traj_3d.png" \
    --step_size 10 \
    --resolution 1920 1080 \
    --frustum_size 0.03

  echo "[${ITER_ID}][4/7] Visualizing lifted NPZ trajectory in top-down scene..."
  CUDA_VISIBLE_DEVICES=1 python visualize_generated_trajectory_topdown.py \
    --semantic_metadata_json "$SEMANTIC_METADATA_JSON" \
    --generated_traj_npz "$LIFTED_NPZ_CANON" \
    --source_traj_txt "$SOURCE_TRAJ_TXT" \
    --intrinsics_path "${DATA_DIR}/pred_intrinsics.txt" \
    --scene_ply_dir "${DATA_DIR}/all_points" \
    --output_dir "$TOPDOWN_NPZ_VIZ_OUTPUT_DIR" \
    --fps "$FPS" \

  echo "[${ITER_ID}][5/7] Visualizing generated cameras in top-down scene (TUM)..."
  CUDA_VISIBLE_DEVICES=1 python visualize_generated_trajectory_topdown.py \
    --semantic_metadata_json "$SEMANTIC_METADATA_JSON" \
    --generated_traj_txt "$LIFTED_TUM_CANON" \
    --source_traj_txt "$SOURCE_TRAJ_TXT" \
    --intrinsics_path "${DATA_DIR}/pred_intrinsics.txt" \
    --scene_ply_dir "${DATA_DIR}/all_points" \
    --output_dir "$TOPDOWN_VIZ_OUTPUT_DIR" \
    --fps "$FPS" \

  echo "[${ITER_ID}][6/7] Rendering scene from lifted cameras..."
  CUDA_VISIBLE_DEVICES=1 python render_scene_from_vlm_cameras.py \
    --data_dir "$DATA_DIR" \
    --traj_path "$LIFTED_TUM_CANON" \
    --output_dir "$RENDER_OUTPUT_DIR" \
    --fps "$FPS"

  if [[ ! -f "${RENDER_OUTPUT_DIR}/vis-videos/render.mp4" ]]; then
    echo "Missing render video output: ${RENDER_OUTPUT_DIR}/vis-videos/render.mp4" >&2
    exit 1
  fi

  TOPDOWN_VIDEO_PATH="${TOPDOWN_VIZ_OUTPUT_DIR}/generated_topdown_trajectory.mp4"
  if [[ "$SKIP_CRITIC" == "0" ]]; then
    echo "[${ITER_ID}][7/7] Evaluating generated trajectory with critic..."
    CRITIC_CMD=(
      python -m agents.critic
      --model "$CRITIC_MODEL"
      --user-demand "$USER_DEMAND"
      --strided-trajectory-json "$VLM_STRIDED_TRAJECTORY_JSON"
      --render-video "${RENDER_OUTPUT_DIR}/vis-videos/render.mp4"
      --critic-out "$CRITIC_RESULT_CANON"
      --reasoning-out "$CRITIC_REASONING_CANON"
      --inputs-dir "$CRITIC_INPUTS_DIR"
      --num-frames "$NUM_FRAMES"
    )
    if [[ -f "$TOPDOWN_VIDEO_PATH" ]]; then
      CRITIC_CMD+=(--topdown-video "$TOPDOWN_VIDEO_PATH")
    else
      echo "Topdown video not found at ${TOPDOWN_VIDEO_PATH}; critic will run with render-only evidence."
    fi
    CUDA_VISIBLE_DEVICES=1 "${CRITIC_CMD[@]}"
  else
    echo "[${ITER_ID}][7/7] Skipping critic step (--skip-critic)."
    mkdir -p "$CRITIC_OUTPUT_DIR"
    if [[ -n "$CRITIC_RESULT_JSON" ]]; then
      if [[ ! -f "$CRITIC_RESULT_JSON" ]]; then
        echo "Missing --critic-result-json for --skip-critic: $CRITIC_RESULT_JSON" >&2
        exit 1
      fi
      cp "$CRITIC_RESULT_JSON" "$CRITIC_RESULT_CANON"
    else
      cat > "$CRITIC_RESULT_CANON" <<'JSON'
{
  "status": "skipped",
  "reason": "Critic evaluation skipped by --skip-critic",
  "intent_matched": false,
  "overall_quality_score": 0.0
}
JSON
    fi
    printf '%s\n' "N/A (critic skipped)" > "$CRITIC_REASONING_CANON"
    mkdir -p "$CRITIC_INPUTS_DIR"
  fi

  if [[ ! -f "$CRITIC_RESULT_CANON" ]]; then
    echo "Missing critic result JSON: $CRITIC_RESULT_CANON" >&2
    exit 1
  fi
  if [[ ! -f "$CRITIC_REASONING_CANON" ]]; then
    echo "Missing critic reasoning text: $CRITIC_REASONING_CANON" >&2
    exit 1
  fi

  ITER_INTENT="$(python - "$CRITIC_RESULT_CANON" <<'PY'
import json
import sys
data = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
print("true" if bool(data.get("intent_matched", False)) else "false")
PY
)"
  ITER_SCORE="$(python - "$CRITIC_RESULT_CANON" <<'PY'
import json
import sys
data = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
score = data.get("overall_quality_score", 0.0)
try:
    print(float(score))
except Exception:
    print(0.0)
PY
)"

  EXECUTED_ITERATIONS=$((i + 1))
  FINAL_ITER_ID="$ITER_ID"
  PREV_CRITIC_JSON="$CRITIC_RESULT_CANON"
  PREV_STRIDED_TRAJECTORY_JSON="$VLM_STRIDED_TRAJECTORY_JSON"

  if [[ "$LOOP_MODE" == "iterative" ]]; then
    if [[ "$ITER_INTENT" == "true" ]]; then
      STOP_REASON="intent_matched"
      BEST_ITER_ID="$ITER_ID"
      break
    fi
    if python - "$ITER_SCORE" "$SCORE_THRESHOLD" <<'PY'
import sys
score = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if score >= threshold else 1)
PY
    then
      STOP_REASON="score_threshold"
      BEST_ITER_ID="$ITER_ID"
      break
    fi
  else
    if [[ "$SKIP_CRITIC" == "0" && "$ITER_INTENT" == "true" ]]; then
      STOP_REASON="intent_matched"
    elif [[ "$SKIP_CRITIC" == "0" ]] && python - "$ITER_SCORE" "$SCORE_THRESHOLD" <<'PY'
import sys
score = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if score >= threshold else 1)
PY
    then
      STOP_REASON="score_threshold"
    else
      STOP_REASON="single_mode_completed"
    fi
    BEST_ITER_ID="$ITER_ID"
  fi
done

if [[ -z "$FINAL_ITER_ID" ]]; then
  echo "No iterations were executed." >&2
  exit 1
fi
if [[ -z "$BEST_ITER_ID" ]]; then
  BEST_ITER_ID="$FINAL_ITER_ID"
fi
if [[ "$LOOP_MODE" == "iterative" && "$STOP_REASON" == "max_iterations_reached" ]]; then
  BEST_ITER_ID="$FINAL_ITER_ID"
fi

printf '%s\n' "$BEST_ITER_ID" > "${EXPERIMENT_DIR}/best_iteration.txt"
printf '%s\n' "$FINAL_ITER_ID" > "${EXPERIMENT_DIR}/final_iteration.txt"
ln -sfn "iterations/${BEST_ITER_ID}" "${EXPERIMENT_DIR}/final"

BEST_ITER_DIR="${ITERATIONS_DIR}/${BEST_ITER_ID}"
FINAL_STRIDED_TRAJECTORY_JSON="${BEST_ITER_DIR}/01_vlm/strided_camera_trajectory_pixels.json"
FINAL_DENSE_TRAJECTORY_JSON="${BEST_ITER_DIR}/01_vlm/camera_trajectory_pixels.json"
FINAL_REASONING_TXT="${BEST_ITER_DIR}/01_vlm/reasoning.txt"
FINAL_USER_DEMAND_TXT="${BEST_ITER_DIR}/01_vlm/user_demand.txt"
FINAL_LIFTED_JSON="${BEST_ITER_DIR}/02_lift/lifted_camera_trajectory_3d.json"
FINAL_LIFTED_TUM="${BEST_ITER_DIR}/02_lift/lifted_pred_traj.txt"
FINAL_LIFTED_NPZ="${BEST_ITER_DIR}/02_lift/lifted_pred_traj.npz"
FINAL_TOPDOWN_DIR="${BEST_ITER_DIR}/03_topdown_generated_trajectory_viz"
FINAL_TOPDOWN_NPZ_DIR="${BEST_ITER_DIR}/03_topdown_generated_npz_viz"
FINAL_NPZ_VIZ_3D_DIR="${BEST_ITER_DIR}/03_lifted_npz_3d_viz"
FINAL_RENDER_DIR="${BEST_ITER_DIR}/04_render_vlm_cameras"
FINAL_CRITIC_JSON="${BEST_ITER_DIR}/05_critic/critic_result.json"
FINAL_CRITIC_REASONING="${BEST_ITER_DIR}/05_critic/reasoning.txt"
FINAL_CRITIC_INPUTS="${BEST_ITER_DIR}/05_critic/inputs"

MANIFEST_PATH="${EXPERIMENT_DIR}/run_manifest.json"
CREATED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
export MANIFEST_PATH CREATED_AT EXPERIMENT_DIR RUN_ID DATA_DIR VLM_MODEL CRITIC_MODEL USER_DEMAND
export FINAL_STRIDED_TRAJECTORY_JSON FINAL_DENSE_TRAJECTORY_JSON
export FINAL_REASONING_TXT FINAL_USER_DEMAND_TXT
export FINAL_LIFTED_JSON FINAL_LIFTED_TUM FINAL_LIFTED_NPZ FINAL_TOPDOWN_DIR FINAL_TOPDOWN_NPZ_DIR FINAL_RENDER_DIR
export FINAL_NPZ_VIZ_3D_DIR
export FINAL_CRITIC_JSON FINAL_CRITIC_REASONING FINAL_CRITIC_INPUTS
export SEMANTIC_METADATA_JSON SOURCE_TRAJ_TXT FPS DEPTH_MODE CONSTANT_DEPTH SKIP_GENERATE SKIP_CRITIC NUM_FRAMES
export SPARSE_FRAME_DIVISOR
export LOOP_MODE MAX_ITERATIONS SCORE_THRESHOLD STOP_REASON EXECUTED_ITERATIONS BEST_ITER_ID FINAL_ITER_ID

python - <<'PY'
import json
import os
from pathlib import Path


def abs_path(path: str) -> str:
    return str(Path(path).resolve())


def path_obj(path: str) -> dict[str, str]:
    return {"relative": path, "absolute": abs_path(path)}


experiment_dir = Path(os.environ["EXPERIMENT_DIR"])
iterations_dir = experiment_dir / "iterations"
iteration_dirs = sorted([p for p in iterations_dir.glob("iter_*") if p.is_dir()])

iterations = []
for p in iteration_dirs:
    critic_path = p / "05_critic" / "critic_result.json"
    intent = None
    score = None
    if critic_path.exists():
        try:
            data = json.loads(critic_path.read_text(encoding="utf-8"))
            intent = bool(data.get("intent_matched")) if "intent_matched" in data else None
            if "overall_quality_score" in data:
                score = float(data["overall_quality_score"])
        except Exception:
            intent = None
            score = None

    iterations.append(
        {
            "iteration_id": p.name,
            "iteration_dir": path_obj(str(p)),
            "intent_matched": intent,
            "overall_quality_score": score,
            "artifacts": {
                "strided_trajectory_json": path_obj(
                    str(p / "01_vlm" / "strided_camera_trajectory_pixels.json")
                ),
                "dense_trajectory_json": path_obj(
                    str(p / "01_vlm" / "camera_trajectory_pixels.json")
                ),
                "reasoning_txt": path_obj(str(p / "01_vlm" / "reasoning.txt")),
                "user_demand_txt": path_obj(str(p / "01_vlm" / "user_demand.txt")),
                "lifted_json": path_obj(str(p / "02_lift" / "lifted_camera_trajectory_3d.json")),
                "lifted_tum": path_obj(str(p / "02_lift" / "lifted_pred_traj.txt")),
                "lifted_npz": path_obj(str(p / "02_lift" / "lifted_pred_traj.npz")),
                "topdown_generated_trajectory_viz_dir": path_obj(
                    str(p / "03_topdown_generated_trajectory_viz")
                ),
                "topdown_generated_npz_viz_dir": path_obj(
                    str(p / "03_topdown_generated_npz_viz")
                ),
                "lifted_npz_3d_viz_dir": path_obj(
                    str(p / "03_lifted_npz_3d_viz")
                ),
                "render_vlm_cameras_dir": path_obj(str(p / "04_render_vlm_cameras")),
                "critic_result_json": path_obj(str(p / "05_critic" / "critic_result.json")),
                "critic_reasoning_txt": path_obj(str(p / "05_critic" / "reasoning.txt")),
                "critic_inputs_dir": path_obj(str(p / "05_critic" / "inputs")),
            },
        }
    )

manifest = {
    "created_at": os.environ["CREATED_AT"],
    "run_id": os.environ["RUN_ID"],
    "experiment_dir": path_obj(os.environ["EXPERIMENT_DIR"]),
    "vlm_model": os.environ["VLM_MODEL"],
    "critic_model": os.environ["CRITIC_MODEL"],
    "user_demand": os.environ["USER_DEMAND"],
    "data_dir": path_obj(os.environ["DATA_DIR"]),
    "loop": {
        "mode": os.environ["LOOP_MODE"],
        "max_iterations": int(os.environ["MAX_ITERATIONS"]),
        "score_threshold": float(os.environ["SCORE_THRESHOLD"]),
        "stop_reason": os.environ["STOP_REASON"],
        "executed_iterations": int(os.environ["EXECUTED_ITERATIONS"]),
        "best_iteration": os.environ["BEST_ITER_ID"],
        "final_iteration": os.environ["FINAL_ITER_ID"],
    },
    "options": {
        "fps": int(os.environ["FPS"]),
        "depth_mode": os.environ["DEPTH_MODE"],
        "constant_depth": float(os.environ["CONSTANT_DEPTH"]),
        "skip_generate": os.environ["SKIP_GENERATE"] == "1",
        "skip_critic": os.environ["SKIP_CRITIC"] == "1",
        "num_frames": int(os.environ["NUM_FRAMES"]),
        "sparse_frame_divisor": int(os.environ["SPARSE_FRAME_DIVISOR"]),
        "loop_mode": os.environ["LOOP_MODE"],
    },
    "inputs": {
        "semantic_metadata_json": path_obj(os.environ["SEMANTIC_METADATA_JSON"]),
        "source_traj_txt": path_obj(os.environ["SOURCE_TRAJ_TXT"]),
    },
    "artifacts": {
        "strided_trajectory_json": path_obj(os.environ["FINAL_STRIDED_TRAJECTORY_JSON"]),
        "dense_trajectory_json": path_obj(os.environ["FINAL_DENSE_TRAJECTORY_JSON"]),
        "reasoning_txt": path_obj(os.environ["FINAL_REASONING_TXT"]),
        "user_demand_txt": path_obj(os.environ["FINAL_USER_DEMAND_TXT"]),
        "lifted_json": path_obj(os.environ["FINAL_LIFTED_JSON"]),
        "lifted_tum": path_obj(os.environ["FINAL_LIFTED_TUM"]),
        "lifted_npz": path_obj(os.environ["FINAL_LIFTED_NPZ"]),
        "lifted_npz_3d_viz_dir": path_obj(os.environ["FINAL_NPZ_VIZ_3D_DIR"]),
        "topdown_generated_trajectory_viz_dir": path_obj(os.environ["FINAL_TOPDOWN_DIR"]),
        "topdown_generated_npz_viz_dir": path_obj(os.environ["FINAL_TOPDOWN_NPZ_DIR"]),
        "render_vlm_cameras_dir": path_obj(os.environ["FINAL_RENDER_DIR"]),
        "critic_result_json": path_obj(os.environ["FINAL_CRITIC_JSON"]),
        "critic_reasoning_txt": path_obj(os.environ["FINAL_CRITIC_REASONING"]),
        "critic_inputs_dir": path_obj(os.environ["FINAL_CRITIC_INPUTS"]),
        "best_iteration_txt": path_obj(str(experiment_dir / "best_iteration.txt")),
        "final_iteration_txt": path_obj(str(experiment_dir / "final_iteration.txt")),
        "final_symlink": path_obj(str(experiment_dir / "final")),
    },
    "iterations": iterations,
}

Path(os.environ["MANIFEST_PATH"]).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

echo "Pipeline complete."
echo "Experiment root: $EXPERIMENT_DIR"
echo "Loop mode: $LOOP_MODE"
echo "Stop reason: $STOP_REASON"
echo "Executed iterations: $EXECUTED_ITERATIONS"
echo "Best iteration: $BEST_ITER_ID"
echo "Final iteration: $FINAL_ITER_ID"
echo "Sparse 2D trajectory: $FINAL_STRIDED_TRAJECTORY_JSON"
echo "Dense 2D trajectory: $FINAL_DENSE_TRAJECTORY_JSON"
echo "Reasoning: $FINAL_REASONING_TXT"
echo "User demand: $FINAL_USER_DEMAND_TXT"
echo "Lifted 3D JSON: $FINAL_LIFTED_JSON"
echo "Lifted TUM: $FINAL_LIFTED_TUM"
echo "Lifted NPZ: $FINAL_LIFTED_NPZ"
echo "3D lifted NPZ viz: $FINAL_NPZ_VIZ_3D_DIR"
echo "Top-down generated trajectory viz: $FINAL_TOPDOWN_DIR"
echo "Top-down NPZ trajectory viz: $FINAL_TOPDOWN_NPZ_DIR"
echo "Render output: $FINAL_RENDER_DIR"
echo "Critic result JSON: $FINAL_CRITIC_JSON"
echo "Critic reasoning: $FINAL_CRITIC_REASONING"
echo "Critic inputs: $FINAL_CRITIC_INPUTS"
echo "Manifest: $MANIFEST_PATH"
