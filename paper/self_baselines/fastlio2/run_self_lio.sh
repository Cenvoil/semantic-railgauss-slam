#!/bin/bash
# FAST-LIO2 on the self-collected sequence: roscore + node + player.
set -u
ROOT=${SELF_ROOT:-<PROJECT_ROOT>}
CONDA_ROOT=${CONDA_ROOT:-<CONDA_ROOT>}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fastlio2
source "$ROOT/fastlio_ws/devel/setup.bash"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ROS_HOME=/tmp/roshome ROS_LOG_DIR=/tmp/roslog ROS_HOSTNAME=localhost
export LD_LIBRARY_PATH=/tmp/sophus_stub:$LD_LIBRARY_PATH
mkdir -p /tmp/roshome /tmp/roslog
export PLAY_DUR=${PLAY_DUR:-1e9}
export PLAY_START=${PLAY_START:-0}
export PLAY_IMG=0
export ODOM_TOPIC=${ODOM_TOPIC:-/Odometry}
export LIVO_TUM=${LIVO_TUM:-$ROOT/self_livo/fastlio_traj_tum.txt}
BIN=$ROOT/fastlio_ws/devel/lib/fast_lio/fastlio_mapping
rm -f "$LIVO_TUM"
roscore > "$ROOT/self_livo/fastlio_roscore.log" 2>&1 &
for i in $(seq 1 90); do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep 2
done
rosparam load "$ROOT/self_livo/self_lio.yaml"
$BIN __name:=laserMapping > "$ROOT/self_livo/fastlio_node.log" 2>&1 &
NP=$!
sleep 5
python3 "$ROOT/self_livo/player3d.py" > "$ROOT/self_livo/fastlio_player.log" 2>&1
sleep 8
kill $NP 2>/dev/null
pkill -f fastlio_mapping 2>/dev/null
pkill -f 'bin/roscore' 2>/dev/null
echo FASTLIO_DONE
