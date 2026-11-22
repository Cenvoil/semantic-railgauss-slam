#!/bin/bash
# Build ORB-SLAM3 with a modern CMake (policy compatibility shim).
set -e
ROOT=${ORB_ROOT:-<PROJECT_ROOT>/benchmarks/repos/ORB_SLAM3}
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
FLAG="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
cd "$ROOT"

rm -rf Thirdparty/DBoW2/build Thirdparty/g2o/build Thirdparty/Sophus/build build

echo "=== DBoW2"
cd Thirdparty/DBoW2 && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release $FLAG > /tmp/orb_dbwo2_cfg.log 2>&1
make -j8
cd "$ROOT"

echo "=== g2o"
cd Thirdparty/g2o && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release $FLAG > /tmp/orb_g2o_cfg.log 2>&1
make -j8
cd "$ROOT"

echo "=== Sophus"
cd Thirdparty/Sophus && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release $FLAG > /tmp/orb_sophus_cfg.log 2>&1
make -j8 || true
cd "$ROOT"

echo "=== vocabulary"
cd Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd "$ROOT"

echo "=== ORB_SLAM3"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release $FLAG > /tmp/orb_main_cfg.log 2>&1
make -j8
echo BUILD_OK
