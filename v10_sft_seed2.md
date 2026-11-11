# v10 — 最佳 C 换 seed 复跑

> 设置：200 iters，lr=1e-5，scale=20，seed=20261111。

promo accuracy=62.00%，parse_fail=19.00%，composite=0.5725。

| 指标 | Base | SFT |
|---|---:|---:|
| 原 held-out 450 | 60.00% | 62.89% |
| 外部中文 200 | 53.50% | 59.50% |
| 混合 650 | 58.00% | 61.85% |

## 分析

- 本组用于验证 C 的收益是否能在新 seed 下复现，不增加数据。
- 与 C 的 62.89% 原 held-out、59.54% 混合成绩对照解读。

## 统计检验与最终结论

- 原 held-out：D 比 Base 高 2.89pp，与 C 的 62.89% 完全复现。
- 外部中文：D 比 Base 高 6.00pp。
- 混合：D 比 Base 高 3.85pp。
- 合计 McNemar：Base 对/D 错 72 题，Base 错/D 对 97 题，p=0.0649。尚未达到 0.05 显著性门槛，但方向性收益在新 seed 下复现。
- D 是目前最优 SFT 配置：200 iters、lr=1e-5、scale=20。
