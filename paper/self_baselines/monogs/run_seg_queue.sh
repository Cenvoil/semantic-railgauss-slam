#!/bin/bash
# Run one queue of MonoGS segments on one GPU.
# usage: run_seg_queue.sh <gpu> <queue_file> <tag>
set -u
GPU=$1
QUEUE=$2
TAG=$3
BASE=${MONOGS_ROOT:-<PROJECT_ROOT>/monogs_run}
PY=${MONOGS_PY:-<CONDA_ROOT>/envs/gigaslam/bin/python}
TOOLS=$(cd "$(dirname "$0")" && pwd)
LOG=$BASE/seg_${TAG}.log
cd "$BASE" || exit 1
while read -r S E; do
  [ -z "$S" ] && continue
  NAME=self_${S}_${E}
  echo "=== $NAME start $(date +%F_%T)" >> "$LOG"
  if [ ! -f "dataset/$NAME/rgb.txt" ]; then
    $PY "$TOOLS/make_monogs_seg.py" --start "$S" --end "$E" >> "$LOG" 2>&1
  fi
  cat > "configs/mono/tum/$NAME.yaml" <<EOF
inherit_from: "configs/mono/tum/base_config.yaml"

Results:
  save_trj: True
  eval_rendering: False

Dataset:
  dataset_path: "$BASE/dataset/$NAME/"
  Calibration:
    fx: 2371.3075
    fy: 2371.3075
    cx: 700.2272
    cy: 364.5387
    k1: 0.022928
    k2: 0.0
    p1: 0.0
    p2: 0.0
    k3: 0.0
    width: 1280
    height: 720
    distorted: True
EOF
  CUDA_VISIBLE_DEVICES=$GPU $PY slam.py --config "configs/mono/tum/$NAME.yaml" >> "$LOG" 2>&1
  echo "=== $NAME done $(date +%F_%T) rc=$?" >> "$LOG"
done < "$QUEUE"
echo "QUEUE_${TAG}_DONE" >> "$LOG"
