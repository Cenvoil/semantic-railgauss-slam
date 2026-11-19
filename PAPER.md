# Paper-aligned numbers

The canonical numbers for the paper are in `paper/results/`:

- `three_sequence.csv` — main v11 results: 1.2 = 1.84 cm, 3.1 = 5.45 cm, 5.1 = 2.94 cm
  (Horn SE(3), w2c-t, mean, INS initialization); PSNR / MS-SSIM / LPIPS / mIoU as reported.
- `ablation_2x2.csv` — 2×2 factorial: 23.45 / 3.41 / 2.57 / 2.22 cm, plus the pure-INS oracle.
- `self_collected_3d.csv` — self-collected sequence, 3D-only PSNR 27.21 dB.
- `s2_three_sequence.csv`, `s2_seed_replication.csv` — per-Gaussian semantic head on LIVO-GS.

The results table in the top-level `README.md` is an **earlier L7 development
configuration** (1.2 ≈ 2.22/2.16 cm; 5.1 ≈ 3.38 cm) and uses different protocol labels
(for example, KISS-ICP is reported in camera-center form). It is retained for
development history and **must not be compared directly with the paper**. For any
paper comparison, use the CSVs in `paper/results/` and the configuration registry in
the paper's Appendix A.

The five numbers 1.84 / 2.22 / 3.41 / 4.27 / 2280.88 cm refer to different runs:

| Number | Run | Meaning |
|---:|---|---|
| 1.84 | main v11 (INS) | paper main table |
| 2.22 | 2×2 factorial Yes/Yes | fixed-rasterizer factorial |
| 3.41 | 2×2 factorial Yes scale / No INS | factorial, no INS |
| 4.27 | main-system no-INS | consolidated no-INS run |
| 2280.88 | main-system no-INS on 5.1 | negative result, no INS |
