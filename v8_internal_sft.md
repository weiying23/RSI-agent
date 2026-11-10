# test-RSI 实验记录 v8 —— 内部自生成 1,500 条 RSR-lite SFT

> 记录时间：2026-11-10 08:29:22
> 训练数据：1350 条 RSR-lite 自生成中文题
> 目标：验证同域数据量是否影响 LoRA SFT
> 训练：LoRA iters=500, lr=5e-5

---

## 1. 结果

| 指标 | Base | SFT |
|---|---:|---:|
| 原 held-out 450 | 60.00% | 47.78% |
| 外部中文 200 | 53.50% | 49.00% |
| 混合 650 | 58.00% | 48.16% |

## 2. 数据规模

- RSR-lite 尝试：2525 条
- 接受：1500 条
- train：1350 条
- valid：150 条

## 3. 结论

见最终分析。

## 4. 文件

- summary：`arms_sft/arm_W_internal_1500/summary.json`
- 日志：`logs/run_internal_sft.log`
