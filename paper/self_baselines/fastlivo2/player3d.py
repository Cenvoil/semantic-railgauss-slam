#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Play the self-collected sequence (3D LiDAR clouds from lidar_all) into FAST-LIVO2."""
import csv
import os
import struct
import threading
import time

import cv2
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField

ROOT = os.environ.get("SELF_ROOT", "<PROJECT_ROOT>")
LID = os.environ.get("SELF_LIDAR3D", ROOT + "/selfcollected/lidar_all")
FMAP = os.environ.get("SELF_LMAP", LID + "/frame_lidar_map.csv")
IMU = os.environ.get("SELF_IMU", ROOT + "/self_livo/imu.npz")
FRAME_DIR = os.environ.get("SELF_FRAMES", ROOT + "/selfcollected/yolo_input/frames")
FRAME_MAP = os.environ.get("SELF_FMAP", ROOT + "/selfcollected/yolo_input/frame_map.csv")
SPEED = float(os.environ.get("PLAY_SPEED", "1.0"))
START = float(os.environ.get("PLAY_START", "0"))
DUR = float(os.environ.get("PLAY_DUR", "1e9"))
USE_IMG = os.environ.get("PLAY_IMG", "0") == "1"
LOOKAHEAD = int(os.environ.get("PLAY_LOOKAHEAD", "10"))

DT = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
    ("t", "<u4"), ("reflectivity", "<u2"), ("ring", "u1"),
    ("ambient", "<u2"), ("range", "<u4"),
])
FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("t", 16, PointField.UINT32, 1),
    PointField("reflectivity", 20, PointField.UINT16, 1),
    PointField("ring", 22, PointField.UINT8, 1),
    PointField("ambient", 23, PointField.UINT16, 1),
    PointField("range", 25, PointField.UINT32, 1),
]

odom_n = [0]
last_stamp = [None]


def load_scans():
    path = os.environ.get("SELF_LIDAR3D_BIN", ROOT + "/self_livo/lidar3d.bin")
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            npts, stamp = struct.unpack("<Id", f.read(12))
            xyz = np.frombuffer(f.read(npts * 12), dtype="<f4").reshape(npts, 3).copy()
            out.append((stamp, xyz))
    return out


def load_frames():
    rows = []
    with open(FRAME_MAP, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), os.path.join(FRAME_DIR, r["file"])))
    return rows


def lidar_msg(stamp, xyz):
    m = PointCloud2()
    m.header.stamp = rospy.Time.from_sec(float(stamp))
    m.header.frame_id = "livox_frame"
    m.height = 1
    m.width = xyz.shape[0]
    m.fields = FIELDS
    m.is_bigendian = False
    m.point_step = DT.itemsize
    m.row_step = DT.itemsize * xyz.shape[0]
    m.is_dense = True
    a = np.zeros(xyz.shape[0], dtype=DT)
    a["x"] = xyz[:, 0]
    a["y"] = xyz[:, 1]
    a["z"] = xyz[:, 2]
    a["intensity"] = 100.0
    a["t"] = 0
    a["reflectivity"] = 100
    a["ring"] = 0
    a["ambient"] = 0
    a["range"] = 0
    m.data = a.tobytes()
    return m


def main():
    rospy.init_node("self_player", anonymous=False)
    pub_lidar = rospy.Publisher("/livox/lidar", PointCloud2, queue_size=50)
    pub_imu = rospy.Publisher("/livox/imu", Imu, queue_size=20000)
    pub_img = rospy.Publisher("/camera/image_color", Image, queue_size=20)

    traj_f = open(os.environ.get("LIVO_TUM", ROOT + "/self_livo/livo_traj_tum.txt"), "w")
    wall_f = open(os.environ.get("SELF_SCANS_WALL_OUT", ROOT + "/self_livo/scans_wall.txt"), "w")
    done = threading.Event()

    def odom_cb(msg):
        odom_n[0] += 1
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            traj_f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" %
                         (msg.header.stamp.to_sec(), p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            traj_f.flush()
        except ValueError:
            pass

    rospy.Subscriber(os.environ.get("ODOM_TOPIC", "/aft_mapped_to_init"), Odometry, odom_cb, queue_size=1000)
    scans = load_scans()
    z = np.load(IMU)
    t_i, w_i, a_i = z["t"], z["w"], z["a"]
    print("loaded scans %d imu %d" % (len(scans), len(t_i)), flush=True)
    rospy.sleep(2.0)

    t0 = float(t_i[0])
    wall0 = time.time() + 0.5

    def due(t):
        return wall0 + (t - t0 - START) / SPEED

    def imu_worker():
        i = 0
        while i < len(t_i) and not rospy.is_shutdown():
            t = float(t_i[i])
            if t - t0 < START or t - t0 > START + DUR:
                i += 1
                continue
            horizon = (last_stamp[0] if last_stamp[0] is not None else (t0 + START)) + 0.35
            if t > horizon:
                time.sleep(0.003)
                if done.is_set():
                    break
                continue
            m = Imu()
            m.header.stamp = rospy.Time.from_sec(t)
            m.header.frame_id = "livox_frame"
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = [float(v) for v in w_i[i]]
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = [float(v) for v in a_i[i]]
            m.orientation_covariance[0] = -1.0
            pub_imu.publish(m)
            i += 1
        print("imu done %d" % i, flush=True)

    threading.Thread(target=imu_worker, daemon=True).start()

    if USE_IMG:
        frames = load_frames()

        def img_worker():
            for t_img, path in frames:
                if t_img - t0 < START or t_img - t0 > START + DUR:
                    continue
                d = due(float(t_img)) - time.time()
                if d > 0:
                    time.sleep(d)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                m = Image()
                m.header.stamp = rospy.Time.from_sec(float(t_img))
                m.header.frame_id = "camera"
                m.height, m.width = img.shape[:2]
                m.encoding = "bgr8"
                m.is_bigendian = 0
                m.step = img.shape[1] * 3
                m.data = img.tobytes()
                pub_img.publish(m)
                if done.is_set():
                    break

        threading.Thread(target=img_worker, daemon=True).start()

    sent = 0
    for stamp, xyz in scans:
        if stamp - t0 < START or stamp - t0 > START + DUR:
            continue
        d = due(float(stamp)) - time.time()
        if d > 0:
            time.sleep(d)
        t_gate = time.time()
        while sent > 40 and sent > odom_n[0] + LOOKAHEAD and not rospy.is_shutdown():
            if time.time() - t_gate > 5.0:
                break
            time.sleep(0.01)
        pub_lidar.publish(lidar_msg(stamp, xyz))
        last_stamp[0] = float(stamp)
        wall_f.write("%.9f %.6f\n" % (float(stamp), time.time()))
        wall_f.flush()
        sent += 1
        if sent % 200 == 0:
            print("lidar %d odom %d" % (sent, odom_n[0]), flush=True)
    print("DONE lidar %d odom %d" % (sent, odom_n[0]), flush=True)
    done.set()
    rospy.sleep(20)
    traj_f.close()
    wall_f.close()


if __name__ == "__main__":
    main()
