# v11 — LoRA scale=10 正则化实验

> 设置：200 iters，lr=1e-5，scale=10，其他与 C 相同。

promo accuracy=58.00%，parse_fail=19.00%，composite=0.5325。

| 指标 | Base | SFT |
|---|---:|---:|
| 原 held-out 450 | 60.00% | 62.44% |
| 外部中文 200 | 53.50% | 52.00% |
| 混合 650 | 58.00% | 59.23% |

## 分析

- scale 从 20 降为 10，目的是减小权重漂移并改善格式稳定性。
- 选择指标为 accuracy − 0.25 × parse_fail_rate。

## 最终结论

- E 的原 held-out 为 62.44%，外部 52.00%，混合 59.23%，均低于 D。
- scale=10 并未带来比 scale=20 更好的格式稳定性，后续不采用。
