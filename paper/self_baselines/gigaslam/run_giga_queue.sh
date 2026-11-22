#!/bin/bash
# Run one queue of GigaSLAM segments on one GPU.
# usage: run_giga_queue.sh <gpu> <queue_file> <tag>
set -u
GPU=$1
QUEUE=$2
TAG=$3
BASE=${GIGASLAM_ROOT:-<PROJECT_ROOT>/gigaslam_osdar}
PY=${GIGASLAM_PY:-<CONDA_ROOT>/envs/gigaslam/bin/python}
TOOLS=$(cd "$(dirname "$0")" && pwd)
LOG=$BASE/seg_${TAG}.log
cd "$BASE" || exit 1
while read -r S E; do
  [ -z "$S" ] && continue
  NAME=self_${S}_${E}
  echo "=== $NAME start $(date +%F_%T)" >> "$LOG"
  $PY "$TOOLS/make_giga_seg.py" --start "$S" --end "$E" --motion 2.0 --loop 0 >> "$LOG" 2>&1
  GIGA_GPU=$GPU $PY slam.py --config "configs/$NAME.yaml" >> "$LOG" 2>&1
  echo "=== $NAME done $(date +%F_%T) rc=$?" >> "$LOG"
done < "$QUEUE"
echo "QUEUE_${TAG}_DONE" >> "$LOG"
