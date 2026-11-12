# 调研文档

论文 PDF 均已下载至本目录，索引如下（下载日期：2026-11-11；精读笔记见 [精读笔记/](精读笔记/) 目录，每篇含"问题/创新/方法/结果/局限"五节，关键数字均与原文核对）。

| # | 论文 | arXiv | 本地文件 | 精读笔记 |
|---|------|-------|----------|----------|
| 1 | RSI-Forge: From Research Papers to Environments for Recursive Self-Improvement | [2610.09426](https://arxiv.org/abs/2610.09426) | `2610.09426_RSI-Forge.pdf` | [RSI-Forge 笔记](精读笔记/RSI-Forge_2610.09426.md) |
| 2 | Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine | [2610.22192](https://arxiv.org/abs/2610.22192) | `2610.22192_Terminal-RSI.pdf` | [Terminal-RSI 笔记](精读笔记/Terminal-RSI_2610.22192.md) |
| 3 | ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents（OpenMBM） | [2611.04998](https://arxiv.org/abs/2611.04998) | `2611.04998_ForgeKernel.pdf` | [ForgeKernel 笔记](精读笔记/ForgeKernel_2611.04998.md) |
| 4 | EmoRSS: Mitigating Emotion-Induced Over-Refusal in Large Language Models | [2610.04998](https://arxiv.org/abs/2610.04998) | `2610.04998_EmoRSS.pdf` | [EmoRSS 笔记](精读笔记/EmoRSS_2610.04998.md) |

`text/` 目录为 pdftotext 提取的全文纯文本（`-layout` 版式），可用于检索引用原文。

## 分析与方案

- [RSI后续实践方案.md](RSI后续实践方案.md) —— 基于 v1–v13 改进链 + 四篇论文精读 + HANDOFF §10 评测宪法（C1–C5）的后续 RSI 实践路线：闭合 D×H₅ 回路、教训库知识回流、swap test 补 C4 证据。

## 摘要速览

### 1. RSI-Forge（2610.09426）
多智能体流水线，把已发表论文转化为可执行的自提升（recursive self-improvement）环境：三个 agent 分别负责构建、复现、评审，产出带自动评估器的任务，并独立复现论文方法作为基线。共构建 210 个环境覆盖 18 个领域（其中 90 个经人类领域专家评审）。在 120 个环境上让四个模型连续 3 轮尝试（继承代码与笔记、权重固定），84% 的环境中至少有一个模型在首轮后继续提升；在 68/120 个环境上超过复现的论文方法。

### 2. Terminal-RSI（2610.22192）
面向终端（terminal）agent 的递归自我提升：交替进行验证轨迹的 Recursive Self-Rewrite（RSR）与继承 critic 的 Recursive Policy-Refine（RPR）。提出 TB-GAE（跨交互轮衰减、保留 token 级基线）、Family Dense Reward、Success-Prioritized Reward、Warm-up Critic 等。从 Qwen3.8-27B 基座经五阶段得到 Terminal-RSI-27B：Terminal-Bench 4.0 得 19.84%，TB 2.1 得 86.52%，TB 3.0 得 16.95%，TB Hard 得 79.00%。

### 3. ForgeKernel（2611.04998，OpenMBM）
自进化的 kernel 生成系统，解决现有 kernel agent 缺乏跨运行知识积累与架构演化的问题：分析高性能参考实现的硬件行为，把验证过的架构决策蒸馏为可复用的硬件机制知识；将通过正确性与性能验证的 kernel "封存"为下一轮优化的参考。在 NVIDIA Hopper/Blackwell 的 27 个算子、29 种输入配置上 25 个匹配或超过专家参考（几何平均 +10%）；在昇腾 910C 上超越 CANN-Bench S1 决榜全部 8 个 level-4 算子的公开最优（几何平均 +25.1%）。截至 2026-10-30，其公开提交在 CANN-Bench 2.0 三个解决方案榜（910C/950DT/950PR）及 SOL-ExecBench FlashInfer-Bench 赛道均排名第一。

### 4. EmoRSS（2610.04998）
发现情绪表达会系统性增加 LLM 对良性请求的拒绝倾向（过度拒绝）。提出 emotion-guided refusal subspace steering：用线性探针定位拒绝敏感层，从与探针方向对齐的 SAE 特征构造拒绝子空间；用同查询的"普通/带情绪"配对请求估计该子空间内的平均激活偏移，反向注入推理过程，不更新主干参数。在两个 LLM 上于"拒绝有害 / 回答良性"之间取得更优权衡。
