#!/bin/bash
# Run FAST-LIVO2 on the self-collected sequence: launch node + play the sequence.
set -u
ROOT=${SELF_ROOT:-<PROJECT_ROOT>}
CONDA_ROOT=${CONDA_ROOT:-<CONDA_ROOT>}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fastlio2
source "$ROOT/livo_gs_ws/devel/setup.bash"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ROS_HOME=/tmp/roshome ROS_LOG_DIR=/tmp/roslog ROS_HOSTNAME=localhost
export LD_LIBRARY_PATH=/tmp/sophus_stub:$LD_LIBRARY_PATH
mkdir -p /tmp/roshome /tmp/roslog
export PLAY_SPEED=${PLAY_SPEED:-1.0}
export PLAY_DUR=${PLAY_DUR:-1e9}
export PLAY_START=${PLAY_START:-0}
export PLAY_IMG=${PLAY_IMG:-0}
export LIVO_TUM=${LIVO_TUM:-$ROOT/self_livo/livo_traj_tum.txt}
export LAUNCH=${LAUNCH:-mapping_self3d.launch}
export PLAYER=${PLAYER:-$ROOT/self_livo/player3d.py}
rm -f "$LIVO_TUM"
roslaunch fast_livo_gs "$LAUNCH" > "$ROOT/self_livo/livo_run.log" 2>&1 &
LP=$!
for i in $(seq 1 90); do
  if rosnode list >/dev/null 2>&1; then
    echo "master up after ${i}0x2s" >> "$ROOT/self_livo/livo_run.log"
    break
  fi
  sleep 2
done
sleep 5
python3 "$PLAYER" > "$ROOT/self_livo/player.log" 2>&1
sleep 8
kill $LP 2>/dev/null
pkill -f fastlivo_mapping 2>/dev/null
pkill -f mapping_self 2>/dev/null
echo RUN_DONE
