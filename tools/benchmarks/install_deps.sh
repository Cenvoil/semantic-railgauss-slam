#!/usr/bin/env bash
# ORB-SLAM3 依赖安装（Pangolin + Eigen，无需 root，装到用户目录）
# 授权：Gary 2026-10-08
#
# 现状：服务器有 g++ 13.3 / cmake 3.28 / 网络可用
#      但无 Pangolin、OpenCV(系统级)、Eigen
#      conda 环境里有 OpenCV 4.13（可复用，见 run_orb3.sh 的 CMake 参数）
#      sudo 需密码 ⇒ 全部装到 ~/local
set -eu

PREFIX=$HOME/local
mkdir -p "$PREFIX/src" "$PREFIX/lib" "$PREFIX/include"
cd "$PREFIX/src"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------- Eigen（header-only，最快）----------
if [ ! -f "$PREFIX/include/eigen3/Eigen/Dense" ]; then
  log "下载 Eigen 3.4.0"
  for url in \
    "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz" \
    "https://github.com/eigenteam/eigen-git-mirror/archive/3.4.0.tar.gz"; do
    if wget -q --show-progress -O eigen.tar.gz "$url" 2>&1 | tail -1; then
      log "  从 $url 下载成功"; break
    fi
  done
  tar xzf eigen.tar.gz
  cd eigen-3.4.0
  mkdir -p build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX" > /dev/null
  make -j4 install > /dev/null
  cd "$PREFIX/src"
  log "Eigen 安装完成"
else
  log "Eigen 已存在，跳过"
fi

# ---------- Pangolin ----------
if [ ! -d "$PREFIX/include/Pangolin" ]; then
  log "克隆 Pangolin v0.6"
  git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git
  cd Pangolin
  mkdir -p build && cd build
  # 关掉视频/OpenGL 相关的可选依赖，减少编译量与失败面
  cmake .. \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF \
    -DENABLE_FFMPEG=OFF \
    -DENABLE_GLCORE=OFF > /dev/null
  make -j4 install > /dev/null
  cd "$PREFIX/src"
  log "Pangolin 安装完成"
else
  log "Pangolin 已存在，跳过"
fi

log "=== 安装结果 ==="
ls "$PREFIX/include/eigen3/Eigen/Dense" && echo "  Eigen OK"
ls "$PREFIX/include/Pangolin/Pangolin.hpp" && echo "  Pangolin OK"
ls "$PREFIX/lib/"*.so* 2>/dev/null | head -5
du -sh "$PREFIX"
