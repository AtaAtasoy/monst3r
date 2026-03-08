#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./visualize_topdown_all.sh --iter-dir ITER_DIR [options]

Wrapper to render:
  - VLM lifted TUM top-down trajectory
  - VLM lifted NPZ top-down trajectory
  - Gen3C top-down trajectory (if gen3c_output is present)

If no variant-specific path arguments are provided, defaults are inferred from ITER_DIR:
  - 02_lift/lifted_pred_traj.txt
  - 02_lift/lifted_pred_traj.npz
  - 02_lift/gen3c_output (optional)

Common options are forwarded to visualize_trajectory_topdown_variants.py.

Examples:
  ./visualize_topdown_all.sh --iter-dir demo_tmp/.../iterations/iter_000
  ./visualize_topdown_all.sh --iter-dir demo_tmp/.../iterations/iter_000 --fps 25
  ./visualize_topdown_all.sh demo_tmp/.../iterations/iter_000 --skip-gen3c
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ITER_DIR=""
ARGS=()
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --iter-dir)
      ITER_DIR="$2"
      shift 2
      ;;
    *)
      if [[ "$1" != --* ]] && [[ -z "$ITER_DIR" ]] && [[ -d "$1" ]]; then
        ITER_DIR="$1"
        shift
      else
        ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ -z "$ITER_DIR" ]]; then
  echo "Missing --iter-dir (or positional iteration directory)." >&2
  usage
  exit 1
fi

if [[ ! -d "$ITER_DIR" ]]; then
  echo "Missing iteration directory: $ITER_DIR" >&2
  exit 1
fi

python "$SCRIPT_DIR/visualize_trajectory_topdown_variants.py" \
  --iter-dir "$ITER_DIR" \
  "${ARGS[@]}"
