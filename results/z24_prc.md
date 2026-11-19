# Z24 real-data PRC (first bending mode, FDD)

calibration: 15 healthy setups, threshold tau = 0.0183 (DKW alpha=5%, delta=5%; n small, so tau = max)
held-out healthy: n=12, FPR at tau = 0.17, median S = 0.0122
all damage scenarios pooled: n=126, TPR at tau = 0.45, AUROC vs held-out healthy = 0.665 (95% bootstrap CI 0.53-0.79)

| scenario | state | mean f1 (Hz) | median S | TPR at tau | AUROC vs held-out healthy |
|---|---|---|---|---|---|
| 1 | reference | 3.914 | 0.0187 | 0.50 | - |
| 2 | pier-lowering system installed | 3.872 | 0.0025 | 0.00 | - |
| 3 | pier lowered 20 mm | 3.869 | 0.0044 | 0.00 | 0.241 |
| 4 | pier lowered 40 mm | 3.857 | 0.0103 | 0.11 | 0.407 |
| 5 | pier lowered 80 mm | 3.763 | 0.0583 | 1.00 | 1.000 |
| 6 | pier lowered 95 mm | 3.684 | 0.0971 | 1.00 | 1.000 |
| 7 | foundation tilt | 3.839 | 0.0197 | 0.56 | 0.843 |
| 8 | second reference | 3.853 | 0.0122 | 0.00 | - |
| 9 | spalling of concrete | 3.868 | 0.0042 | 0.00 | 0.204 |
| 10 | spalling (larger) | 3.862 | 0.0078 | 0.00 | 0.287 |
| 11 | landslide at abutment | 3.851 | 0.0111 | 0.11 | 0.491 |
| 12 | concrete hinge failure | 3.848 | 0.0156 | 0.33 | 0.611 |
| 13 | anchor-head failure 1 | 3.845 | 0.0161 | 0.33 | 0.722 |
| 14 | anchor-head failure 2 | 3.842 | 0.0184 | 0.56 | 0.806 |
| 15 | tendon rupture 1 | 3.845 | 0.0163 | 0.33 | 0.722 |
| 16 | tendon rupture 2 | 3.829 | 0.0262 | 1.00 | 0.981 |
| 17 | tendon rupture 3 | 3.827 | 0.0253 | 1.00 | 0.991 |

noise-only floor (within-scenario std of f1, relative): 0.0017 -> omega^2 floor ~0.0034; calibrated threshold 0.0183 is 5.4x that floor (healthy-state spread across scenarios dominates; temperature is not controlled).
