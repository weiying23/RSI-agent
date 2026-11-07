# test-RSI 实验记录 v6 —— 均衡中文 SFT

> 记录时间：2026-11-08
> 目的：用“自生成 verified 中文题 + 翻译后中文 GSM8K 对齐题”做均衡 SFT，验证外部数据能否提升权重通道。
> 训练：LoRA rank8/scale20，iters=300，lr=5e-5
> 数据：50% 自生成 verified / 50% 中文 GSM8K 对齐题

---

## 1. 结果

| 评测集 | 准确率 |
|---|---:|
| 原 held-out 前 100 题 | **47.0%** |
| heldout_zh（200 道中文 GSM8K） | **39.5%** |

对照组：

```text
BASE_checked               = 57.0%
harness-only arm_H         = 66.0%
旧 arm_W SFT               = 54.0%
arm_W + hints 首代         = 59.0%
本次均衡中文 SFT           = 47.0%（原 held-out）
```

## 2. 结论

- 均衡中文 SFT 没有超过基线；
- 比 harness-only 差 19pp；
- 比旧 arm_W SFT 还低 7pp；
- 在中文 GSM8K 上的泛化只有 39.5%。

## 3. 分析

1. 外部中文 GSM8K 和自生成题库仍不同域：GSM8K 是生活应用题，自生成题是公式化数学题；
2. 训练数据总量仍小（285 条训练 / 31 条验证）；
3. LoRA 训练 300 iters 可能过拟合了混合分布；
4. 评估侧无提示，而训练数据一部分带 hints，存在训练/评估不一致；
5. 当前 3B + LoRA 在这个任务上，权重更新仍难以产生稳定收益。

## 4. 文件

- 训练集：`data/sft_balanced_v1/`
- adapter：`arms_sft/arm_W_balanced_v1/adapters`
- 汇总：`arms_sft/arm_W_balanced_v1/summary.json`
- 翻译脚本：`rsi/translate_gsm8k.py`
