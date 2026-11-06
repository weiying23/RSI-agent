# test-RSI 实验记录 v5 —— MLX GRPO-Lite LoRA 小试

> 记录时间：2026-11-05
> 目标：回答“把 SFT 换成 RL，权重通道是否改善”
> 实现：MLX 上极简 REINFORCE/GRPO-Lite，LoRA 8/scale20
> 设置：3 轮，每轮 pool=20，group_size=4，epochs=2，lr=1e-5
> 评测：held-out 前 100 题，runtime_mode=checked

---

## 1. 结果

| 轮次 | rollout 正确率 | held-out sys |
|---|---:|---:|
| 1 | 47/80 (58.8%) | **31%** |
| 2 | 28/80 (35.0%) | **28%** |
| 3 | 28/80 (35.0%) | **0%** |

对照：

```text
BASE_checked     = 57%
arm_W SFT 最终    = 54%
arm_H harness    = 66%
arm_W_RL 第1轮   = 31%
arm_W_RL 第3轮   = 0%
```

## 2. 发生了什么

- RL 第 1 轮就让 held-out 从 57% 跌到 31%；
- 第 3 轮模型彻底失去 `\boxed{}` 输出格式，`parse_fail_rate = 1.0`；
- rollout 正确率也从 58.8% 降到 35%。

## 3. 结论

在这套“小模型 + LoRA + 学术题 + 0/1 奖励”的极简 RL 下，权重通道比 SFT 更差：

> **RL 没有改善权重通道，反而更快地导致格式和能力坍缩。**

主要问题不是 RL 概念无效，而是这个极简 GRPO-Lite 缺少：

- 格式奖励 / KL 惩罚；
- 足够多的 on-policy 样本；
- 更稳的 advantage 估计；
- 防止模型忘记基础行为的约束。

因此当前最强结论仍然是：

> **在 3B + LoRA + 学术题规模下，harness-only 是最优路线；权重 SFT 和权重 RL 都没有正向收益。**

## 4. 文件

- RL 脚本：`rsi/rl_grpo.py`
- RL 运行目录：`arms_rl/arm_W_RL/`
- 日志：`logs/rl_arm_W_RL_v2.log`
