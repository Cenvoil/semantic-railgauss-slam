#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""口径回归测试：验证 eval_ate.py 能复现项目已知的数字。

为什么必须做：`eval_ate.py` 是 benchmarks 工程所有指标的唯一出口。
如果它算不出项目自己的 2.22 cm，"同口径"就是空话。

验证对象（项目权威实现 utils/eval_helpers.py 跑出来的）
    d8_gk1045           3.41 cm
    d8_insinit2.57 cm
    d8_gk1045_insinit   2.22 cm

做法：同一个 npz，分别用项目 evaluate_ate 与我们的 compute() 算，必须一致。

用法: python regression_test_ate.py
"""
import os
import sys

import numpy as np

ROOT = '<PROJECT_ROOT>/SGS-SLAM'
EXP = os.path.join(ROOT, 'experiments', 'Railway_d')
BENCH = '<PROJECT_ROOT>/benchmarks'

EXPECT = {
    'd8_gk1045': 3.41,
    'd8_insinit': 2.57,
    'd8_gk1045_insinit': 2.22,
}

# 5.1 序列（里程 101.48 m，是 1.2 的 4.8倍——最适合暴露口径 bug）
EXPECT_51 = {
    'd8_51_insinit': 3.38,
}


def quat2R(q):
    """SGS-SLAM 存储的四元数是 wxyz。"""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def test_synthetic(ea):
    """合成数据自检：用已知真解造数据，看能不能恢复出来。

    ★ 这是最有效的防bug 手段：真实数据里残差"看起来合理"，
    但只有已知真解才能证明公式方向/顺序完全正确。
    2026-10-08 实战：两个 bug（平移多乘scale、旋转用 @R.T）
    在真实轨迹上只表现为"Sim3 比 SE3 大"这个模糊现象，
    靠合成数据才立刻定位到方向错了。
    """
    print('-' * 74)
    print('附加自检：合成数据（已知真解，应精确恢复）')
    ok = True
    np.random.seed(0)
    n = 60
    X = np.random.randn(n, 3)
    ang = 0.3
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0],
                   [np.sin(ang), np.cos(ang), 0],
                   [0, 0, 1]])
    s_true = 0.8
    t_true = np.array([1.0, 2.0, 3.0])

    # 无噪声：应精确恢复
    Y = (s_true * X @ Rz.T) + t_true
    r, t, sc = ea.align_sim3(X, Y)
    res = np.linalg.norm(ea.apply_align(X, r, t, sc) - Y, axis=1).mean()
    good = (abs(sc - s_true) < 1e-6 and np.allclose(t, t_true, atol=1e-6)
            and res < 1e-6)
    ok = ok and good
    print('无噪声 Sim3: scale %.8f(真 %.4f) 残差 %.2e  %s'
          % (sc, s_true, res, 'OK' if good else 'FAIL'))

    # SE(3) 用**无尺度**的合成数据（SE(3) 本就不该拟合带尺度的数据）
    Y_rigid = X @ Rz.T + t_true
    r2, t2, s2 = ea.align_se3(X, Y_rigid)
    res2 = np.linalg.norm(ea.apply_align(X, r2, t2, s2) - Y_rigid,
                          axis=1).mean()
    good2 = res2 < 1e-6 and np.allclose(t2, t_true, atol=1e-6)
    ok = ok and good2
    print('无噪声 SE3 (刚体数据): 残差 %.2e  %s'
          % (res2, 'OK' if good2 else 'FAIL'))

    # 有噪声：Sim3 残差必须 <= SE3 残差
    Yn = Y + np.random.randn(n, 3) * 0.01
    m_se3 = ea.compute(Yn, X)
    good3 = m_se3['ate_sim3_mean_cm'] <= m_se3['ate_cm'] + 1e-6
    ok = ok and good3
    print('带噪声: SE3 %.6f cm  Sim3 %.6f cm  %s'
          % (m_se3['ate_cm'], m_se3['ate_sim3_mean_cm'],
             'OK' if good3 else 'FAIL (Sim3 > SE3，对齐公式有错)'))
    return ok


def test_sim3_le_se3(ea):
    """自检：Sim(3) 的残差必须 <= SE(3)。

    为什么需要这个断言：早期版本 `align_sim3` 把 scale 提前乘进了 trans，
    应用时又乘一次 scale，导致平移被放大 scale 倍，
    Sim(3) 残差反而比 SE(3) 大两个数量级（1.2 序列上测出 883 cm vs 7cm）。
    ★ 这类bug 靠人眼看数字不容易发现，必须有断言。
    """
    print('-' * 74)
    print('附加自检：Sim(3) 残差应 <= SE(3) 残差')
    ok = True
    for run in EXPECT:
        path = os.path.join(EXP, run, 'params.npz')
        if not os.path.exists(path):
            continue
        p = np.load(path, allow_pickle=True)
        rots = p['cam_unnorm_rots'][0]
        trans = p['cam_trans'][0]
        n = rots.shape[1]
        gt_raw = np.asarray(p['gt_w2c_all_frames'])[:n]
        est_np = np.zeros((n, 3))
        gt_np = np.zeros((n, 3))
        for i in range(n):
            est_np[i] = trans[:, i]
            gt_np[i] = np.asarray(gt_raw[i])[:3, 3]

        m = ea.compute(gt_np, est_np)
        good = m['ate_sim3_mean_cm'] <= m['ate_cm'] + 1e-6
        ok = ok and good
        print('%-24s SE3 %7.2f cm  Sim3 %7.2f cm  %s'
              % (run, m['ate_cm'], m['ate_sim3_mean_cm'],
                 'OK' if good else '!! Sim3 > SE3，align_sim3 有 bug'))
    return ok


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(BENCH, 'scripts'))
    import torch
    from utils.eval_helpers import evaluate_ate
    import eval_ate as ea

    rows = []
    for run, expect in EXPECT.items():
        path = os.path.join(EXP, run, 'params.npz')
        if not os.path.exists(path):
            print('skip %s (no params.npz)' % run)
            continue
        p = np.load(path, allow_pickle=True)
        rots = p['cam_unnorm_rots'][0]
        trans = p['cam_trans'][0]
        n_kf = rots.shape[1]
        gt_raw = np.asarray(p['gt_w2c_all_frames'])[:n_kf]

        est_t, gt_t = [], []
        est_np = np.zeros((n_kf, 3))
        gt_np = np.zeros((n_kf, 3))
        for i in range(n_kf):
            T = np.eye(4)
            T[:3, :3] = quat2R(rots[:, i])
            T[:3, 3] = trans[:, i]
            est_t.append(torch.tensor(T, dtype=torch.float64))
            gt_t.append(torch.tensor(np.asarray(gt_raw[i]), dtype=torch.float64))
            est_np[i] = T[:3, 3]
            gt_np[i] = np.asarray(gt_raw[i])[:3, 3]

        ref = evaluate_ate(gt_t, est_t) * 100      # 项目权威
        mine = ea.compute(gt_np, est_np)['ate_cm']  # 我们的实现

        rows.append((run, ref, mine, expect))

    #---- 5.1 序列同样回归（里程长，最容易暴露口径 bug）----
    for run, expect in EXPECT_51.items():
        path = os.path.join(EXP, run, 'params.npz')
        if not os.path.exists(path):
            continue
        p2 = np.load(path, allow_pickle=True)
        r2 = p2['cam_unnorm_rots'][0]
        t2 = p2['cam_trans'][0]
        n2 = r2.shape[1]
        gt2 = np.asarray(p2['gt_w2c_all_frames'])[:n2]
        est_t2, gt_t2 = [], []
        est_np2 = np.zeros((n2, 3))
        gt_np2 = np.zeros((n2, 3))
        for i in range(n2):
            T = np.eye(4)
            T[:3, :3] = quat2R(r2[:, i])
            T[:3, 3] = t2[:, i]
            est_t2.append(torch.tensor(T, dtype=torch.float64))
            gt_t2.append(torch.tensor(np.asarray(gt2[i]), dtype=torch.float64))
            est_np2[i] = T[:3, 3]
            gt_np2[i] = np.asarray(gt2[i])[:3, 3]
        ref2 = evaluate_ate(gt_t2, est_t2) * 100
        mine2 = ea.compute(gt_np2, est_np2)['ate_cm']
        rows.append((run + ' (5.1)', ref2, mine2, expect))
        print('%-24s project=%.4f  ours=%.4f  diff=%.4f'
              % (run, ref2, mine2, abs(ref2 - mine2)))

    print('=' * 74)
    print('ATE 口径回归测试  (主口径 = SE(3) + mean + w2c_t)')
    print('=' * 74)
    print('%-24s %10s %10s %9s %9s' % ('run', '项目权威', '本脚本', '差', '冻结值'))
    print('-' * 74)
    ok = True
    for run, ref, mine, expect in rows:
        d = abs(ref - mine)
        if d >= 0.01:
            ok = False
        print('%-24s %9.4f %10.4f %9.4f %8.2f  %s'
              % (run, ref, mine, d, expect, 'OK' if d < 0.01 else 'MISMATCH'))
    print('-' * 74)

    ok = test_synthetic(ea) and ok
    ok = test_sim3_le_se3(ea) and ok

    print('=' * 74)
    print('结论: %s' % ('PASS 口径与项目一致，同口径对比成立'
                        if ok else 'FAIL 口径有偏差，不可用于对比'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())