# v14 —— 回到 RSI：权重提升点 × harness 进化路线

> 时间：2026-11-12  
> 当前最佳权重：D  
> 当前最佳 harness：H₅  
> 目标：在统一 650 题混合测试集上超过 66%

---

## 1. 我们已经拿到的“权重提升点”

### 1.1 当前最佳 SFT 配置

| 项目 | 设置 |
|---|---|
| 权重 | **D** |
| 数据 | `data/sft_eval_format_v1` |
| train / valid | 1,350 / 150 |
| LoRA iters | 200 |
| learning rate | 1e-5 |
| LoRA scale | 20 |
| seed | 20261111 |
| 训练格式 | H0 system + 裸题 user + 直接推理 assistant |

### 1.2 D 的结果

| 指标 | Base | D |
|---|---:|---:|
| 原 held-out 450 | 60.00% | **62.89%** |
| 外部中文 200 | 53.50% | **59.50%** |
| 混合 650 | 58.00% | **61.85%** |

### 1.3 已验证有效的权重训练经验

1. **训练/评估格式必须一致**  
   - system 使用 H0；
   - user 只给题目；
   - assistant 直接推理并输出 `\\boxed{}`。
2. **200 iters / lr=1e-5 / scale=20 是当前最优组合**。  
   - `scale=10` 没有带来格式稳定，准确率反而下降。
3. **不要再混外部数据**。  
   - 80:20 外部混合在 promo 上好看，但 held-out 显著退化。
4. **选择指标要包含格式稳定性**。  
   - 推荐使用：`accuracy − 0.25 × parse_fail_rate`。
5. **权重提升已经可复现**。  
   - D 在新 seed 下复现了 C 的原 held-out 收益。

---

## 2. 当前最佳 harness：H₅

来自 `arm_H/gen_005`：

| 参数 | 值 |
|---|---|
| temperature | 0.6 |
| top_p | 0.95 |
| max_tokens | 512 |
| retry_k | 3 |
| format_retry | 1 |
| self_consistency_k | 2 |
| k_shots | 2 |
| shot_policy | hard_win |

`arm_H` 在 held-out 前 100 题上达到 **66%**。

注意：这个 66% 和 D 的 61.85% **不是同一测试集规模**，不能直接比较。必须在统一 650 题混合测试集上重新评估。

---

## 3. 回到 RSI 的第一步：D × H₅ 四变体

在统一 650 题混合测试集上评估：

| 变体 | 权重 | Harness | 目的 |
|---|---|---|---|
| Base | 原始 Qwen | H₀ | 基线 |
| `w_only` | **D** | H₀ | 权重单独贡献 |
| `h_core` | 原始 Qwen | H₅，无 few-shot | 指令/参数贡献 |
| `h_full` | 原始 Qwen | H₅，含 2 个 hard-win 例题 | harness 完整贡献 |
| `sys` | **D** | **H₅ 完整** | 权重 + harness 协同 |

核心判据：

```text
sys > max(w_only, h_full)
```

并且：

```text
parse_fail 上升不超过 5pp
```

只有满足这个条件，才说明 D 和 H₅ 有真实协同。

---

## 4. 如果协同成立：启动一代真正的 RSI 共进化

### 4.1 初始化

```text
W₀ = D
H₀ = H₅
```

### 4.2 Phase A：冻结 D，搜索 H₆

probe 使用：

```text
25 道 D 当前失败题 + 25 道固定新题
```

候选包括：

1. 当前 H₅；
2. 两个程序化扰动：
   - few-shot 数量 / 策略；
   - temperature、self_consistency_k、retry；
3. 一个由 D 生成的 self-rewrite 指令。

选择指标：

```text
score = accuracy − 0.25 × parse_fail_rate
```

如果 H₆ 不优于 H₅，保留 H₅，不强行更新。

### 4.3 Phase B：冻结 H₆，训练 W₆

1. 用 `(D, H₆)` 生成新题解；
2. 只保留：
   - 答案正确；
   - lint 通过；
   - 多次采样稳定正确；
3. 转成评估一致格式；
4. 与 D 的干净训练数据累积回放；
5. 从原始基座重新训练 LoRA。

训练配置沿用 D：

```text
200 iters
lr=1e-5
scale=20
```

### 4.4 Phase C：四变体评估

比较：

```text
旧组合：D + H₅
新权重：W₆ + H₀
新 harness：Base + H₆
新组合：W₆ + H₆
```

### 4.5 Phase D：联合晋升与回滚

保存最佳组合，而不是分别保存最佳权重和最佳 harness：

```text
best_pair = (W_best, H_best)
```

晋升条件：

```text
新 sys 比旧 sys 高至少 2pp
且 parse_fail 上升不超过 5pp
```

否则整体回滚到 `(D, H₅)`。

---

## 5. 目标与判定

目标是在统一 650 题混合测试集上达到：

```text
> 66%
```

判定标准：

1. `sys` 超过 `w_only` 和 `h_full`；
2. `parse_fail` 不显著恶化；
3. 至少两个 seed 或 avg@3 复现；
4. 配对 McNemar 检验尽量达到 p < 0.05。

---

## 6. 对 RSI 的启示

- 权重通道现在终于有了可复现的正收益；
- harness 通道仍是总体最强路线；
- RSI 的关键不是“权重变强”或“harness 变强”单独发生，而是二者能否协同；
- 下一步必须先验证 `D + H₅`，再启动共进化；
- 如果没有协同，继续单独优化 harness，而不是强行双通道。

---

## 7. 下一步执行顺序

```text
1. D × H₅ 四变体评估
2. 若协同成立：
   H₆ 搜索 → W₆ 训练 → 四变体评估 → 联合晋升
3. 若无协同：
   暂停权重 SFT，继续 harness-only
```

