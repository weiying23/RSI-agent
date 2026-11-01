# test-RSI：单机双通道自进化回路（RSI demo）

> 纲领见 [PROGRAM.md](PROGRAM.md)，完整设计与研究结论见 [HANDOFF.md](HANDOFF.md)。
> 本 README 只讲：它是什么、纪律是什么、怎么跑。

## 这是什么

在一台 MacBook（M4 / 16GB）上闭合「权重 × harness」双通道自进化回路：

```
每代（arm_WH）：
  Phase A  harness 搜索（冻结 W_n）── probe 上比 3 个候选（2 程序化扰动 + W_n 自改指令）
  Phase B  数据生成 + 训练（冻结 H_{n+1}）── verified 解 → LoRA → W_{n+1}（fresh from base）
  Phase C  四变体 held-out 评估（lockbox 独立进程）
  Phase D  晋升门 + 归因入档（promo-val，held-out 绝不做选择）
```

三臂对照：`arm_W`（只训权重）｜`arm_H`（只进化 harness＝市面产品缩影）｜`arm_WH`（交替）。

- 定位：Self-HPO（arXiv:2610.22791）Table 1 的 M4/SFT 最小复现 + 四变体归因 + 坍缩消融
- 主张（claim-indexed）：**C1+C2**；C4 以 swap test 消融呈现；不主张 C5

## 评测纪律（不可违反，全文引用为「纪律§n」）

1. **改进者无手**：improver = (W, H) 纯文本进出，无工具、无文件访问
2. **单一判分实现**：`rsi/grade.py` 是唯一算法，`lockbox/*` 只读包装 + SHA256 校验
3. **held-out 三不**：不训练、不做选择、不被循环代码读取——唯一读方是 `lockbox/` 脚本
4. **选择与结论分离**：一切选择在烧毁型数据（probe/pool/promo-val）上做，结论只在 held-out 上出
5. **全轨迹归档**：hack 日志是一等产出，不是 bug 报告
6. **归因优先于提升**：每代四变体归因可解释
7. **回归保护默认开**（晋升门）；「接受回归」仅作消融
8. **统计功效先行**：held-out ≥400 题 + avg@3，否则不启动 P2

数据纪律：一源五角色一账本——`env_gen` 唯一数据源；实例 ID 全局唯一；
`state/ledger/` 记录每个 ID 的角色，永不跨角色复用。

## 目录

```
rsi/        核心包：env_gen / grade / genome / engine / distill / train / evaluate 编排
lockbox/    防篡改评估层：heldout_v1.jsonl(+sha256) / run_eval.py / grade_heldout.py / freeze.py
arms/       三臂世代目录（gen_000…），每代 genome/samples/train/adapters/eval/lineage
state/      config.json（冻结状态）/ calibration.json / ledger/ / baseline/
scratch/    P0 开环 smoke 的临时产物
tests/      单元测试（env_gen 符号自洽 / grade 边界用例）
```

## 运行（用 lora/.venv 的 python）

```bash
PY=/Users/yingwei/Documents/code/lora/.venv/bin/python

# P0（已完成）：标定 → 冻结 → 开环
$PY -m rsi.calibrate --levels 2,3,4,5 --n 36      # 难度标定（一次性）
$PY lockbox/freeze.py --difficulty <d> --n 450     # 冻结 held-out（一次性，人工确认难度后）
$PY -m rsi.p0 --pool 20 --retry 2 --iters 40 --eval-n 60   # 开环 smoke（一条命令验收）
$PY -m rsi.baseline --eval-n 450 --reps 1          # W_base 全量 t=0 锚点

# 测试
$PY -m unittest tests.test_env_gen tests.test_grade
```

P1（单代 dry-run）/ P2（三臂 × 5 代）的编排器见 `rsi/orchestrator.py`（P1 交付）。

## 开发工具与实验对象的区分

用 AI 辅助开发编排器/判分器代码 **不污染实验**：开发工具在改进面之外
（如同 Self-HPO 的作者用电脑写代码不影响其实验纯度）。改进面 = (W, H)：
权重经 LoRA 训练改变，harness 经基因组搜索改变，两者之外的任何东西
（env_gen、grade、held-out、评估协议）在三臂间完全同构、全程冻结。
