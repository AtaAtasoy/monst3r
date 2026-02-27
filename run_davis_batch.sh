#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")"
PY_CMD=(/home/atasoy/miniconda3/bin/conda run -n monst3r python)

data_root="demo_tmp/davis"
subjects=(bear car-roundabout car-turn horsejump-high mallard-water tennis train)
conf_thresholds=(0.25 0.5 0.75)
distance_percentiles=(0.95 0.80)
shows=(fg bg both all)
look_modes=(origin camera_centroid scene fg)

# Match Python suffix formatting: 0.95 -> 0p95, 0.5 -> 0p5
suffix_from_vals() {
  local dist="$1" conf="$2"
  local dist_fmt conf_fmt
  dist_fmt=$(printf "%.15g" "$dist" | sed 's/-/m/g; s/\./p/g')
  conf_fmt=$(printf "%.15g" "$conf" | sed 's/-/m/g; s/\./p/g')
  echo "_pct${dist_fmt}_conf${conf_fmt}"
}

for subj in "${subjects[@]}"; do
  base="${data_root}/${subj}"

  for dist in "${distance_percentiles[@]}"; do
    for conf in "${conf_thresholds[@]}"; do
      suffix=$(suffix_from_vals "$dist" "$conf")

      echo "[filter] ${subj} dist=${dist} conf=${conf}"
      "${PY_CMD[@]}" vlm_prep_scripts/filter_points_advanced.py \
        "$base" \
        --distance_percentile "$dist" \
        --conf_threshold "$conf" \
        --normalize \
        --copy_images

      render_dir="${base}/normalized${suffix}"

      for show in "${shows[@]}"; do
        for look in "${look_modes[@]}"; do
          echo "[render] ${subj} dist=${dist} conf=${conf} show=${show} look=${look}"
          "${PY_CMD[@]}" render_top_down.py \
            --data_dir "$render_dir" \
            --resolution 1024 \
            --projection perspective \
            --show "$show" \
            --look_at_mode "$look"
        done
      done
    done
  done

done
