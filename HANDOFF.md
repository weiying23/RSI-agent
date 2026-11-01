# test-RSI 项目交接文档（HANDOFF）

> 本文档由上一会话于切换目录前写就，是新会话的唯一上下文入口。
> 新会话第一步：**通读本文档** → 检查文末"当前进度" → 向用户确认接续点。

---

## 0. 给新会话的引导指令

1. 你在用户的 Mac 上（Apple M4，10 核，16GB RAM，macOS），通过 DSH Desktop Beta 运行。
2. 目标目录：`/Users/yingwei/Documents/code/agent/test-RSI`（已存在，当前为空）。**新会话应以它为工作区**（用户已为此换会话）。
3. 先读完本文档，再回应用户。用户用中文交流。

## 1. 任务与用户

- **任务**：研究 RSI（Recursive Self-Improvement）并给出一个 demo。
- 用户核心论点（已验证成立，是整个研究的出发点）：市面绝大多数"RSI"停留在 agent 优化自己的 skills/harness 的范畴（脚手架层），而不是"改进产物回流改进者"的彻底自进化闭环。
- **协作风格**：用户重视概念精确性。曾纠正过我一次错误——把 DeepSeek Super-agent 架构按"形式相似"过度织入 RSI 叙事（正确结论：自主性≠自改进，正交轴）。**勿重犯：映射要按功能，不按形状；结论要经得起用户的追问。**

## 2. 已确立的研究结论（概念层，全部经过用户确认）

1. **RSI 判据 = 回路闭合性**：设改进系统 I_t（权重+数据管线+harness+底座），产出 O_t。伪 RSI：O_t 不回流，改进率有上界（冻结模型天花板）。真 RSI：O_t 以受控方式重新构成 I_{t+1}，判据是"改进率的改进率非零"。
2. **L0–L4 分级**：L0 脚手架（prompt/memory/skill/tool/workflow）｜L1 数据层（agent 选数据，人按训练键）｜L2 权重闭环（自生成+验证→微调→新权重成为下一代生成器，STaR/Self-Rewarding LM/Absolute Zero 在此档但外环人写死）｜L3 元层（agent 改改进过程本身，Darwin Gödel Machine/Gödel Agent 在此档但底层 LLM 冻结）｜L4 架构层。真 RSI = L2+L3，公开系统无人兼有。
3. **BAAI kernel 综述**（arXiv:2601.15727，"Towards Automated Kernel Generation in the Era of LLMs"）§7 明确把 "long-horizon agentic training + autonomous harness engineering" **并列为未来方向**——双通道无人同时做，这就是 demo 的立足点（空白的第一个微型实证）。
4. **Anthropic AAR**（automated-alignment-researchers，2026-04）：9×Claude Opus + 沙箱/论坛/评分服务器，weak-to-strong PGR 人类 0.23 → AAR 0.97（5 天/$18k）。关键教训：①评估是第一瓶颈（"瓶颈从生成转向评估"）；②reward hacking 必然出现（数学题选众数、代码题跑测试读答案），靠人识别取消资格；③泛化是改进与过拟的分界（held-out 0.94 数学/0.47 代码，生产规模 Claude Sonnet 4 上无统计显著提升）；④种群多样性重要，过度结构化有害。
5. **DeepSeek Super-agent**（微信公众号《Super-agent 设计探索》）：自主性/并发/自改进三根轴正交。它是前两根轴，零自改进。仅三个窄接触点：内部实例参与 Harness 自身迭代（一句话）；Watchdog 可在线改主系统提示词/触发规则（运行时可变=必要条件非充分条件）；异步交互提高人审闸门带宽（二阶）。**不要再把它织进 RSI 叙事。**
6. **模型坍缩**（Shumailov et al., Nature 2024）：自生成数据递归训练会自噬，需新熵（外生环境流）对抗。
7. **三铁律**：评估是第一瓶颈且必须防篡改（评估层在改进面之外，是不动点）；泛化是分界线；轨迹（失败+诊断）比产物值钱。

## 3. Demo 完整设计（架构层，已定稿）

**目标**：在一台 M4 上闭合 (W,H) 双通道循环，四变体归因可测，三臂对照出曲线。

### 3.1 三臂
- arm_W：只进化权重（H 冻结在 H_0）
- arm_H：只进化 harness（W 冻结在基座）＝市面产品缩影
- arm_WH：双通道交替 = 空白组合，主角
- 三臂共用同一 engine / 同一 grade / 同一 lockbox / 同一 held-out（改进面外的东西必须同构，归因才可比）

### 3.2 每代协议（arm_WH；W/H 臂是其子集）
```
Phase A  harness 搜索（冻结 W_n）
  probe = 30 新题（用后烧毁）
  candidates = 程序化参数扰动×2 + W_n 自己改写 instruction×1（harness 通道的改进者=当前权重→耦合）
  评分语义 = pass@retry_k（数据产率，下游是 Phase B）＋准确率并列记录
  H_{n+1} = argmax；全部候选+分数入档案
Phase B  数据生成+训练（冻结 H_{n+1}）
  pool = 200 新题（用后烧毁）
  samples = engine(W_n, H_{n+1}).generate(pool, k=retry_k)
  verified = 判对 ∧ lint过 ∧ 每题只留最短正确解
  train_data = gen_1..n 累积（STaR 式回放）→ fresh LoRA 从基座重训 → W_{n+1}
  90/10 切分：90% train，10% valid.jsonl（仅 MLX loss 监控，禁入任何结论）
Phase C  四变体评估（独立进程读 lockbox）
  sys     = (W_n, H_n 完整)
  w_only  = (W_n, H_0, k_shots=0)   纯权重，无示例混淆
  h_core  = (W_base, H_n 关闭 few-shot) 纯指令+参数
  h_full  = (W_base, H_n 含 few-shot)    含技能库
  h_full − h_core = 技能库 co-adaptation 通道净效应
  评估期生成全部弃置
Phase D  归因+档案：Δ四变体、tokens、wallclock、hack_flags；回归接受不回滚（守门模式留作消融）
```

### 3.3 基因组（L0 改进面）
```json
{"instruction":"...要求逐步推理,答案写\\boxed{}",
 "k_shots":0-3, "shot_policy":"none|recent_success|hard_win",
 "temperature":0.3-1.0, "top_p":0.8-0.95, "max_tokens":256-512, "retry_k":1-4}
```
H_0 = {k_shots:0, 默认指令, temp 0.7} ＝"市面出厂设置"。few-shot 示例池来自历代 verified 解（属于 H 的状态，随权重代际变好→co-adaptation 载体）。

### 3.4 数据池设计（一源五角色一账本）
唯一数据源 = env_gen（程序化中文数学题+符号计算答案，无 LLM）。

| 池 | 大小 | 生命周期 | 用途 | 永不接触 |
|---|---|---|---|---|
| 标定集 | ~60 | P0 用完烧毁 | 难度旋钮标定基座到 held-out 50–70% | 之后一切 |
| held-out v1 | 100 | t=0 冻结+SHA256 | 唯一结论性评估，三臂共享 | 所有循环代码 |
| probe_gen_n | 30/代 | 用后烧毁 | Phase A 候选打分 | 训练/示例池/held-out |
| pool_gen_n | 200/代 | 用后烧毁 | Phase B 数据生成 | held-out/下一代 |
| verified 累积 | ≤200/代 | 跨代累积 | 90% 训练 + 10% valid + 示例池 | held-out |
| 示例池 | 累积 | 跨代 | k_shots>0 时 engine 注入 | 训练标签/判分 |

**防污染四层防线**：①改进者无手（improver=(W,H) 纯文本进出，无工具无文件访问）；②单一判分实现（grade.py 唯一算法，lockbox 只读包装+校验和）；③排除账本（实例 ID=模板+参数哈希，ledger.json 记角色并全局排除重用；held-out 三臂共享，probe/pool 账本按臂分）；④评估期生成弃置+SHA256 审计。
**诚实局限**：held-out 是实例级不相交≠模板级不相交，测的是分布内泛化；跨模板 held-out v2 留作 P3 消融（预期缩水，对应 AAR"0.97→生产无效"同款现象）。

### 3.5 六个关键设计决策
1. 四变体评估（原三变体升级：h 拆 core/full，把技能库耦合单独测出）
2. STaR 式 fresh adapter + 累积回放（续训 adapter 留作消融：代际继承 vs 档案回放）
3. 环境发生器外置 = 新熵来源（无新题模式留作消融，预期展示坍缩）
4. probe 轮换 + held-out SHA256（防 harness 过拟 probe、防泄漏）
5. 每题只留最短正确解 + lint（单一 \boxed、最短推理长度、无退化）防风格坍缩
6. 回归接受不回滚，诚实曲线（守门模式留作消融）

### 3.6 结果预注册（演示前写好，四分支都有话讲）
①WH > max(W,H) 协同成立；②WH≈H 任务被 harness 饱和→调难；③WH≈W 指令变异算子太弱→加强；④W 崩溃→去重失效，坍缩本身是发现。

### 3.7 目录骨架
```
test-RSI/
├── rsi/{env_gen,genome,engine,grade,distill,evaluate,train,archive,orchestrator}.py
├── lockbox/{heldout_v1.jsonl(+sha256), grade_heldout.py}
├── arms/{arm_W,arm_H,arm_WH}/gen_NNN/{genome.json,samples.jsonl,train.jsonl,adapters/,eval.json,lineage.json}
├── report/plot.py
├── status.py   # 外部 watchdog：卡代/超时检测
└── README.md
```

### 3.8 算力预算（M4 实测依据）
- 主推模型 Qwen2.5-3B-Instruct-4bit（**已在 ~/.cache，1.6GB，零下载**；1.5B 未缓存，如需需设 HF_HOME 到可写目录或用户终端下载）
- 每代 ≈330K tokens；WH 臂 ≈2.5h，W 臂 ≈2h，H 臂 ≈0.5h → 一代三臂 ≈5h → 5 代 ≈25h（≈2–3 通宵）
- 裁剪旋钮：4 代/150 题/retry_k=2 可减半。/tmp 空闲 101GB。
- P0 难度标定：先跑基座在标定集上的准确率，调 env_gen 难度旋钮到 50–70% 再冻结 held-out。

### 3.9 分阶段
- **P0（半天）**：env_gen+grade+ledger+lockbox+engine 跑通；难度标定；held-out 冻结。验收：一条命令跑通"生成→判分→蒸馏→训练→held-out 评估"开环。
- **P1（1 天）**：arm_WH 单代 dry-run（gen_000→001）。验收：lineage 四变体齐全、档案可查、断点续跑可用。
- **P2（1–2 晚）**：三臂 × 5 代全量 + plot.py 三张图（主曲线/归因瀑布/成本）。
- **P3（可选）**：消融（续训 adapter / 无去重坍缩 / 守门 / 跨模板 held-out v2 / STaR rationalization）。

## 4. 环境事实（新会话直接可用）

- **机器**：Apple M4，10 核，16GB RAM。盘 228GB，空闲 101GB。
- **沙箱**：DSH 会话只能写「会话工作区根 + /tmp」，读全局。执行 ps/sudo 被禁。网络有代理（env 已配）。**本会话教训：若工作区= test-RSI 则可写该目录。**
- **用户 MLX 流水线**（复用来源）：`/Users/yingwei/Documents/code/lora/`
  - `config.yaml`：mlx_lm lora，Qwen2.5-3B-Instruct-4bit，rank 8/scale 20，iters 1000，mask_prompt: true，data: ./data（train.jsonl/valid.jsonl，messages 格式）
  - `.venv/`：已装 mlx-lm + datasets（可直接用其 python 跑 mlx）
  - `data/train.jsonl` 格式：`{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}`
  - 复用：训练 config 核心、messages 格式、load()+generate() 模式、.venv。退役：prepare_data.py（HF 外部教师数据=回路断裂，恰是 demo 要替换的头号项）、verify_data.py。
- **模型**：mlx-community/Qwen2.5-3B-Instruct-4bit 已缓存（snapshots 完整：model.safetensors 等）。
- /tmp 会被 macOS 周期清理（重启后可能丢），重要成果要及时搬入 test-RSI 或打 tar。

## 5. 当前进度（新会话从这里接）

> **2026-11-01 P0 开工更新**：新会话（工作区=test-RSI）已按本节执行，详见下述。
> - ✅ 设计定稿（本文档 §3 即完整设计，勿重新设计，直接实现）
> - ✅ 目标目录：/Users/yingwei/Documents/code/agent/test-RSI（本会话工作区，可写）
> - ✅ 目录骨架 + 全部核心模块：rsi/{config,ledger,env_gen,grade,genome,engine,distill,train,calibrate,p0,baseline,orchestrator}.py + lockbox/{common,freeze,run_eval,grade_heldout}.py + status.py
> - ✅ 单元测试 40 例全过（env_gen 符号自洽 ×10模板×5难度；grade 边界用例）
> - ✅ 引擎实测：3B-4bit 离线加载 2.7s，独占 GPU 时 ~40 tok/s
> - ✅ 难度标定（判分器带分数 bug 修复后重跑）：d2=91.7% / d3=77.8% / d4=72.2% / **d5=61.1% ✓ 落带**
> - ✅ **held-out v1 已冻结（2026-11-01，用户确认难度 5）**：450 题 + SHA256 `483c0e78…`，模板分布 44–46/模板
> - ✅ **p0 开环 smoke 跑通**（14.7 min，2026-11-01）：`python -m rsi.p0` 一条命令完成 生成→判分→蒸馏→训练→held-out 评估。BASE 48.3% → SMOKE 63.3%（60 题前缀，12 训练样本 × 40 iters），配对 13进/4退，McNemar p≈0.049；**parse_fail 16.7%→8.3% = 格式失败被蒸馏修复的首个证据**（brain-body 通道）
> - ✅ **W_base 全量 baseline（t=0 锚点）= 61.33%**（450 题全量，与标定 61.1% 一致，稳落 [50–70%] 带）。基线 hack 指标：parse_fail 9.3%（最大的免费提升空间）、multi_boxed 0.4%、no_reasoning 0.2%。逐模板：travel 89%…frac_addsub 13%（天花板/地板结构健康）
> - ✅ **P0 验收全部达成（2026-11-01）**。下一步 = P1：`python -u -m rsi.orchestrator --arm WH --gens 1 --gen-size tiny --eval-n 60 --reps 1` 做单代 dry-run
> - P2 算力现实（实测 45.8 tok/s 独占）：reps=1 时 WH 臂/代 ≈3.5h、W ≈1.5h、H ≈1.5h → 三臂×5代 ≈32h；纪律§8 的 avg@3 若逐代执行 ≈66h。**建议（待用户定）**：逐代 reps=1 + 配对检验（McNemar，450 题配对比二项 σ 强得多，smoke 已验证）+ 末代 avg@3 审计——保统计功效的同时把 P2 压回 ~35h
> - P0 后新增工程改进：Phase C 变体按臂裁剪+签名去重（arm_W 从 4 变体降到 1，省 ~5h/代）；run_eval 断点续跑（响应复用+重新判分）；标定样本全量归档；GPU 严格串行（并行会互相拖慢 10 倍）
> - **P0 发现与修复记录**：
>   1. NO_PROXY 含 `[::1]` 崩 httpx → config.sanitize_proxy_env() + HF_HUB_OFFLINE=1
>   2. 判分器带分数正则无分隔符 → 假分数 17/15 被拆成 1又7/15（分子≥2位全污染）→ 修复+回归测试；首轮流化数据作废重跑
>   3. d5 折扣模板参数空间仅 30 组合 + 采样器按成功数轮转致活锁 → 扩空间（z∈4.5–9.5，P≤2000）+ 按尝试数轮转+耗尽移除
>   4. **GPU 严格串行**：16GB M4 上两个模型进程并行会互相拖慢 10 倍——P2 三臂必须顺序执行
>   5. 模型真实失败模式（保留，不修）：算对但不写 \boxed{}（格式失败=脑体错配信号，正是 harness/权重双通道该学的东西）；三数 lcm 算术错
> - 实现要点偏离记录：训练数据 messages 不含 system（权重学行为、指令属 harness，归因干净）；held-out 已按 §9.3 扩至 450 题；arm_H 的 Phase C 中 sys≡h_full 别名处理防重复生成；标定样本全量归档（state/calibration_samples_d*.jsonl）

## 6. 操作备忘

- 用户搬文件习惯：Finder，Cmd+Shift+G 输路径，拖拽。给出明确指令即可。
- 过夜全量跑：建议用户在自己 Terminal 跑（不受 DSH 沙箱限制），加 `caffeinate -is` 防合盖睡眠。
- 评估纪律三条（写进 README）：改进者无工具；单一判分实现；held-out 只有 lockbox 一个读方。
- 文献锚点：kernel 综述 arXiv:2601.15727；AAR anthropic.com/research/automated-alignment-researchers（2026-04）；Super-agent 微信文章《Super-agent 设计探索：迈向更自主的 Agent 系统》（原 DeepSeek Harness 团队，Yujia Qiao）；Shumailov et al. Nature 2024（model collapse）。

---

## 7. 扩展调研与判断修正（2026-10-31 增补，覆盖 §2 部分结论）

> 本节基于对 ~30 篇 self-evolve agent 论文的系统调研（arXiv API 全文检索 + 摘要精读）。
> **它修正了 §2 中的一条重要判断，新会话必须以本节为准。**

### 7.1 判断修正：双通道空白已于 2026-06~10 被填补
§2.3 曾断言"公开系统无人同时做 L2+L3"。**此判断已过时**，2026 年出现明确的双通道系统：
- **OPD-Evolver**（arXiv:2606.17628，2026-06）：快环=记忆驱动的 test-time 进化；慢环=on-policy 自蒸馏进策略权重。
- **Self-HPO**（arXiv:2610.22791，2026-10）：H-step（策略分析自身失败、提议 harness 候选）与 P-step（在选定 harness 下 RL 更新策略）交替；**策略自己是 harness 提议者**。与我们的 demo 设计（Phase A/B 交替、W_n 改写 instruction）同构。
- **CoEvo-RSI**（arXiv:2610.25078，2026-10）：模型-harness 协同进化；提出 **brain-body mismatch（脑体错配）** 概念=进化中的 harness 要求模型未学的行为/压制已有能力；共享进化档案 + 检查点选择/合并；**SFT 于自身拒绝采样轨迹**（与我们 verified-only 蒸馏同型）；Qwen3.5-9B。
- **AgentEvolver**（arXiv:2511.10395，2025-11）：self-questioning（任务自生成）+ self-navigating + self-improving（策略 RL）。
- **LM2LM**（arXiv:2610.23344）：RL 训练"提出模型改进方案"的能力本身，research world model 作代理奖励。
- **RSI Arena**（arXiv:2610.33624）：10 个前沿模型当自主 post-training agent（各 1000 GPU-h），阶段 2 加人类反馈。

### 7.2 仍然成立并获强化的结论
1. **用户原始论点获独立实证**：Auditing RSI（arXiv:2610.27610）人工编码 114 篇研究，其中 90 篇自称 RSI：**90% 只做产品评估、60% 用未参与选择的任务评估、仅 22% 审计递归过程本身**。"市面绝大多数 RSI 不具备 RI 能力"从观点升级为统计事实。
2. **三铁律全部获文献印证**：
   - 评估第一瓶颈：VibeTraining-Bench（2610.26342）硬性 safeguard = 测试隔离 + 研究/判分容器分离 + 断网 = 我们的 lockbox 设计已成领域共识。
   - 泛化分界：CFOE（2610.22984）证明 **repair-generalization gap**——失败经验能修复旧任务、几乎不迁移未见任务；成功轨迹才显著迁移。
   - 轨迹值钱：CoEvo-RSI 的共享进化档案、OPD-Evolver 的 memory attribution、AgentEvolver(2610.11613) 的版本化实体。
3. **安全框架**：SAVER（2610.00093）——"经验成为可复用状态后，过去的事件变成未来的原因"。

### 7.3 L0 一线（scaffold/memory/skill，权重冻结）代表性系统
Agent Zero（frdel/agent-zero：2026 版 README 已不再宣传 self-improving，转向"可检查/可编辑的脚手架+插件生态+项目隔离"；其"自进化"=人类策划的 prompt/skill/memory 积累）｜Voyager(2305.16291) 技能库｜Reflexion(2303.11366) 言语 RL｜ExpeL(2308.10144)｜AFlow(2410.10762) MCTS 搜工作流｜ADAS(2408.08435)｜Alita-G(2510.23601) MCP 工具合成｜EvoAgent(2604.20133) 技能+委派｜Living-Harness(2607.26598)｜HarnessPro(2610.26770，注意：LLM 生成测试=弱验证器)｜Skill^∞(2610.35434，脚手架空间内的真递归：元技能重写自身)｜AgentEvolver(2610.11613) 系统级、基座冻结、SWE-bench Pro 82%。

### 7.4 L2 一线（权重闭环、外环写死）
STaR(2203.14465)｜SPIN(2401.01335)｜Self-Rewarding LM(2401.10020，判分能力与指令能力双轴共进)｜SEAL(2506.10943，自编辑=自生成微调数据+更新指令)｜Absolute Zero(2505.03335，任务生成通道内生+代码执行器验证)。

### 7.5 demo 重定位（重要，取代 §0/§3 中的旧叙事）
**旧定位（作废）**："双通道空白的第一个微型实证"——空白已被填。
**新定位（三差异化，全部站得住）**：
1. **归因与审计**：Self-HPO/CoEvo-RSI 报整体提升，未见四变体归因分解；78% 的 RSI 研究不审计过程（audit paper）。demo 的四变体评估+lineage = 对这个 78% 缺口的示范，且 h_core/h_full 分解恰是 CoEvo-RSI "brain-body mismatch" 的可测量化。
2. **单机可复现**：RSI Arena 1000 GPU-h/agent、Self-HPO 需 RL 基建；demo 在一台 M4 用 LoRA-SFT（无 RL）闭合同结构，实现路线与 CoEvo-RSI 的 SFT 版同型（拒绝采样≈我们的 verified-only）。"双通道效应在极小规模是否成立"本身是开放问题。
3. **坍缩动力学**：CFOE 只测了 L0 通道的迁移 gap；双通道的退化/协同过拟合动力学仍是空白 = demo P3 消融的价值。
三臂对应文献线：arm_H = Agent Zero/Voyager/HarnessPro 线的缩影；arm_W = STaR/AZRED 线的缩影；arm_WH = Self-HPO/CoEvo-RSI 线的最小复现。demo = 三条线的最小对照模型。

---

## 8. 调研论文总表（2026-10-31，与 §7 配套）

> 覆盖本轮调研全部对象。2022–2024 经典条目为既有知识+标题核验；2025–2026 条目均经 arXiv API 摘要精读核验（检索窗口至 2026-10-31，共扫 ~40 个标题、精读 ~28 篇摘要）。

### 8.1 L0/L3 线：脚手架/记忆/技能/工作流自进化，权重冻结

| # | Agent/系统 | 年份 | arXiv | 主要 RSI 论点 |
|---|---|---|---|---|
| 1 | Reflexion | 2023-03 | 2303.11366 | 言语强化学习：用语言自我反思替代梯度更新，经验存入情景记忆 |
| 2 | Voyager | 2023-05 | 2305.16291 | 首个 LLM 终身学习具身智能体：可执行代码技能库+自动课程+自验证，GPT-4 黑盒零微调 |
| 3 | ExpeL | 2023-08 | 2308.10144 | 从成功/失败轨迹提炼可迁移 insights 与技能——经验即学习 |
| 4 | EvoAgent | 2024-06 | 2406.14228 | 进化算法自动生成多智能体专家群体（agent 作为可进化个体） |
| 5 | ADAS | 2024-08 | 2408.08435 | 元智能体以代码编程新智能体，把「设计 agent」本身变成 agent 的任务 |
| 6 | AFlow | 2024-10 | 2410.10762 | 工作流优化=代码空间搜索：MCTS+执行反馈自动生成 agentic workflow |
| 7 | SkillWeaver | 2025-04 | 2504.07079 | Web agent 把网站转成 API 技能自我改进——改进=发现并打磨技能 |
| 8 | Alita | 2025-05 | 2505.20286 | 极简预定义的通才智能体：按需自创 MCP 工具，摆脱人工工具预设 |
| 9 | DGM（Darwin Gödel Machine） | 2025-05 | 2505.22954 | 自改自身代码+智能体档案库开放进化；用实证验证替代 Gödel machine 不可行的「证明」 |
| 10 | Alita-G | 2025-10 | 2510.23601 | 从成功轨迹系统化合成/抽象/策展 MCP 工具，由通才变成领域专家 |
| 11 | EvoAgent（二代） | 2026-04 | 2604.20133 | 结构化技能单元+三级记忆+子智能体委派，用户反馈闭环驱动技能进化 |
| 12 | Living-Harness | 2026-07 | 2607.26598 | 轨迹+评估信号→有界 harness 更新（情景记忆+状态图），harness 不再静态 |
| 13 | HarnessPro | 2026-10 | 2610.26770 | 递归 harness 自改进+性能/成本帕累托；无标注路线（LLM 生成测试=弱验证器），产出 lite/medium/heavy 三档 |
| 14 | Skill^∞ | 2026-10 | 2610.35434 | 把「优化流程」编码为元技能，元技能用同一流程优化自身——脚手架空间内的真递归 |
| 15 | Darwin-Claw | 2026-10 | 2610.23574 | 环境自适应经验固化：细粒度错误归因→操作技能与持久记忆协同演化（agent-actor + evolver-critic） |
| 16 | AgentEvolver（系统级） | 2026-10 | 2610.11613 | 8 类实体纳入统一版本化生命周期做系统级进化，基座冻结；SWE-bench Pro 82.08% |
| 17 | Agent Zero | 2025→2026 | GitHub* | prompt 即程序：记忆/技能/层级全在可编辑脚手架；2026 版 README 已撤下 self-improving 叙事，转向可检查/可编辑+插件生态 |
| 18 | Gödel Agent | 2025-07 | ID 未验证* | 运行时自改自身逻辑的自指框架，底层 LLM 冻结 |

\* Agent Zero 为开源框架无论文（github.com/frdel/agent-zero，README 已核验）；Gödel Agent 两次 arXiv 检索未定位 ID，凭先前了解收录，新会话可复核。

### 8.2 L2 线：权重闭环，外环人写死

| # | Agent/系统 | 年份 | arXiv | 主要 RSI 论点 |
|---|---|---|---|---|
| 19 | STaR | 2022-03 | 2203.14465 | 推理自举：只微调导致正确答案的推理；rationalization（给答案提示补推理）破冷启动 |
| 20 | SPIN | 2024-01 | 2401.01335 | 自博弈微调：与历史版本的自己博弈，无需新标注榨干既有数据潜力 |
| 21 | Self-Rewarding LM | 2024-01 | 2401.10020 | 模型自当裁判发奖励做迭代 DPO——指令能力与判分能力双轴共进，通往超人反馈 |
| 22 | Absolute Zero（AZRED） | 2025-05 | 2505.03335 | 零数据自博弈：自提任务（最大化学习进展）+自解任务，代码执行器验证，无外部数据 |
| 23 | SEAL | 2025-06 | 2506.10943 | 自编辑=自生成微调数据+更新指令→持久权重更新；RL 以下游性能为奖励训练「会自编辑」的能力 |

### 8.3 双通道线（2026 起，权重+harness 同时进化）——原「空白」，已被填补

| # | Agent/系统 | 年份 | arXiv | 主要 RSI 论点 |
|---|---|---|---|---|
| 24 | AgentEvolver（初代） | 2025-11 | 2511.10395 | self-questioning（好奇心驱动任务自生成）+ self-navigating + self-improving（策略训练），摆脱人工数据集 |
| 25 | OPD-Evolver | 2026-06 | 2606.17628 | 快慢双环：快环=四级记忆 test-time 进化；慢环=结果校准记忆归因+特权后见蒸馏进策略权重 |
| 26 | LM2LM | 2026-10 | 2610.23344 | 用 RL 训练「提出模型改进方案」的能力本身；冻结的 research world model 作代理奖励 |
| 27 | Self-HPO | 2026-10 | 2610.22791 | harness 与策略交替优化、互相促进：策略自己当 harness 提议者（H-step），在选定 harness 下 RL（P-step），难题反哺下轮搜索 |
| 28 | CoEvo-RSI | 2026-10 | 2610.25078 | 「脑体错配」（brain-body mismatch）：meta-agent 协调 harness 开发+模型检查点选择/合并；SFT 自身拒绝采样轨迹 |
| 29 | RSI Arena | 2026-10 | 2610.33624 | 10 个前沿模型当自主 post-training agent（各 1000 GPU-h/$300）：自选数据/自设计训练/自选检查点；两阶段加人类反馈 |

### 8.4 评估/安全线（三铁律的实证）

| # | Agent/系统 | 年份 | arXiv | 主要 RSI 论点 |
|---|---|---|---|---|
| 30 | CFOE | 2026-10 | 2610.22984 | 修复-泛化鸿沟：失败经验能修复旧任务、几乎不迁移未见任务；成功轨迹才显著迁移 |
| 31 | Continual Learners 研究 | 2026-10 | 2610.25858 | 自进化 agent=非参数持续学习器；梯度更新速度跟不上任务流到达速度 |
| 32 | VibeTraining-Bench | 2026-10 | 2610.26342 | 数据-训练器-harness 跨组件协同设计基准；防作弊三件套=测试隔离+研究/判分容器分离+断网 |
| 33 | Auditing RSI | 2026-10 | 2610.27610 | 114 篇全文本人工编码：90 篇自称 RSI，仅 22% 审计递归过程、60% 用未参与选择的任务评估——「产品≠过程」 |

### 8.5 综述

| # | 文献 | 年份 | arXiv | 核心视角 |
|---|---|---|---|---|
| 34 | A Survey of Self-Evolving Agents | 2025-07 | 2507.21046 | 首个系统综述：按「进化什么（模型/记忆/工具/架构）·何时（测试内/测试间）·怎么」组织领域 |
| 35 | Self-Evolving Agents as Dynamic Graph Transformation | 2026-06 | 2608.18104 | 把自进化 agent 统一为动态图变换的形式视角 |
| 36 | Safety in Self-Evolving Agents（SAVER） | 2026-09 | 2610.00093 | 经验成为可复用状态后「过去的事件变成未来的原因」：安全属性须在累积/泛化/跨上下文复用中存活 |

### 8.6 背景文献（支撑论证的非 agent 系统）

| # | 文献 | 年份 | 来源 | 对本研究的支撑 |
|---|---|---|---|---|
| 37 | Towards Automated Kernel Generation | 2026-01 | arXiv 2601.15727 | §7 将 agentic training 与 harness engineering 并列为未来方向——「双通道空白」判断的出处（该空白 2026-10 已被 §8.3 填补） |
| 38 | Anthropic AAR | 2026-04 | anthropic.com 博客 | 9×Opus 自动对齐研究（PGR 0.23→0.97）：评估第一瓶颈、reward hacking 必现、泛化是改进/过拟合的分界 |
| 39 | Shumailov et al. | 2024-05 | Nature | 递归训练于自生成数据→模型坍缩（自噬）——新熵（外生环境流）必要性的出处 |
| 40 | DeepSeek Super-agent | 2026-10 | 微信公众号 | 自主性≠自改进（正交轴）：系统架构与 RSI 无直接关系，仅三个窄接触点（见 §2.5） |

---

## 9. 主参考论文深读与 demo 设计修正（2026-10-31 第二次增补）

> 精读了 Self-HPO（2610.22791）与 CoEvo-RSI（2610.25078）全文方法+实验。
> **结论：Self-HPO 为主参考（协议层），CoEvo-RSI 为实现层参考。** 以下修正覆盖 §3 对应条目。

### 9.1 Self-HPO 核心机制（协议参考）
- 形式化：agent = (H, P)，目标 J(H,P)=期望解决率；可执行验证器二值评分。
- **H-step（冻结策略）**：当前 (H_t,P_t) 在搜索集 S_t 跑轨迹 → P_t 双重角色（解题者+提议者）分析失败、提出对 prompt/运行时逻辑的针对性修订 → K=6 候选逐一在 S_t 评估、累积经验指导下迭代 → 保留最优 H_{t+1}。
- **P-step（冻结新 harness）**：整个训练集上 GRPO+BAR 强化学习（BAR=自适应补采样构造奖励均衡组，对抗稀疏二值奖励的零方差组）→ P_{t+1}；同时从 rollout 提取「难题集」（全部采样均未解出）= S_{t+1}。
- **搜索集演化**是关键耦合：harness 进化永远瞄准当前权重的失败模式。
- 规模：Qwen3.5-35B-A3B + mini-SWE-agent 种子，2484 SWE 任务，2 轮交替。
- **Table 1（=我们的三臂实验）**：Base 61.0 / 纯策略 RLVR 67.5 / 纯 harness Self-Harness 65.2 / **Self-HPO 69.5**（外部提议者版 71.6）；OOD 泛化（SWE-Multi +5.5pp）。
- **Table 2 消融（对我们最关键）**：
  - **RFT（拒绝采样微调自身成功轨迹）70.5 vs RL 71.6，仅差 1.1pp** → 我们的 verified-only LoRA 蒸馏路线被直接验证可行；
  - 固定搜索集 −1.5pp → 失败条件化 probe 有效；
  - 自提议 −2.1pp（vs Opus 外部提议）→ 闭环纯度只花 2pp 代价。

### 9.2 CoEvo-RSI 核心机制（实现参考）
- 归档驱动双向继承：冻结 meta-agent（Curator/编辑器/模型继承者三个角色策略）协调——诊断失败→检索归档→K 个编辑器各选历史 harness 基座修改→为每个新 harness 做检查点选择或 **TIES 合并**（α 系数由 meta-agent 提议）→ 验证集选对→**拒绝采样+SFT**（与我们的 verified 蒸馏同型）→**晋升门**（gain>margin 才晋级，回归保护）→ 全部结果含失败入归档。
- **TIMR**（回合级接口失配率）：脑体错配的廉价可测诊断。
- 注意纯度妥协：meta-agent 是冻结外部协调者；Self-HPO 才是纯自指闭环。
- 结果：Qwen3.5-9B 协同进化超更大模型；单侧进化大幅落后（harness-only 差 11.49pp）；十轮累积 AA +11.11pp、TIMR≤0.22%。

### 9.3 对 demo 的六条设计修正（新会话按此执行，覆盖 §3.2/§3.4）
1. **probe 失败条件化**：Phase A 的 probe 集 = 上一代 Phase B 中 k 次采样全失败的题 + 新题各半（Self-HPO 证明 +1.5pp）。probe 仍烧毁，但「失败题」可重复充当 probe（它们没进训练数据，不构成泄漏；新题补充防过拟合）。
2. **晋升门设为默认**：W_{n+1} 训练后在**独立晋升验证集**（新增池角色 promo-val，30 新题/代，烧毁）上与 W_n 比较，增益>margin 才晋级，否则保留 W_n（全记录）。**held-out 绝不做晋升门**（选择压力泄漏）。此为 CoEvo-RSI 验证的标准做法；原「回归接受」降为消融。
3. **统计功效修正（关键）**：Self-HPO 双通道效应量仅 2.0–4.3pp（500 题×3 次评估才显著）。我们原定 100 题×1 次，σ≈5pp 测不出。修正：**held-out 扩至 400–500 题**（env_gen 程序生成，成本只在推理 token），四变体评估用 avg@3。
4. **TIES 合并 LoRA adapter 进 P3 消融**：三种权重管理对照——代际继承（续训 adapter）/ 累积回放（fresh from base，默认）/ TIES 合并历史 adapter（CoEvo-RSI 路线）。
5. **格式失配率 = 我们的 TIMR**：\boxed 解析失败率/格式违规率/无推理直答率，作为脑体错配的廉价诊断，进 lineage.json。
6. **叙事最终版**：demo = Self-HPO Table 1 的 M4/SFT 最小复现（其 RFT 消融证明 SFT 路线只损 ~1pp）+ 四变体归因（两篇都没有）+ 坍缩消融 + 失败条件化 probe 与晋升门的完整落地。

---

## 10. RSI 判定与评测的统一框架（2026-10-31 第三次增补，精读 audit+survey 全文所得）

> 本节是项目的「评测宪法」，新会话的评估设计以此为准。来源：Auditing RSI（2610.27610）全文精读 + Survey（2507.21046）§2 形式化定义。

### 10.1 五层主张层级 C1–C5（audit 论文的核心框架）
- **C1 产品主张**：最终系统 > 初始系统（固定评估预算）＝任务增益
- **C2 累积**：增益跨轮累积（保持）
- **C3 可复现**：独立重跑中增益再现
- **C4 改进者增益**：更晚的改进者从同一起点、同预算下产出更好的后代（**独特地"递归"的主张**）——证据=换改进者测试（swap test / paired probe / ignition test）
- **C5 复利**：改进者增益自身随时间增长（swap test 在声明视野上重复）
关键句：「**固定改进者的循环可以让 C1–C3 全部为真**」——迭代改进 ≠ RSI，R 字母的证据几乎无人提供。
实测基线（90 篇自称 RSI）：60% 用 held-out 任务、**22% 审计 ≥2 次全环重跑**；26 篇声称更好改进者，**仅 4 篇做同起点同预算对照**。
四种改变类型：任务内修正 / 持久改变（权重/提示/记忆/工具/harness 代码）/ 结构改变（改改进者）/ 标准改变（改评估者）。

### 10.2 三个命题（评测无效性的形式化根源）
1. **空环基准（winner's curse）**：即使所有候选都不带来真实提升，纯噪声下 N=25 候选也会"接受"约 2.9 个改进、报告出 ≈2.51σ 的虚假增益（N=100 时）；且虚假增益随候选数增长、与轮数无关。→ 对策：密封审计（A2）、**空对照（A5：打乱反馈跑同样回路，直接测虚假增益）**、噪声感知晋升规则（A6，即我们的 margin δ）。
2. **耦合审计继承开发偏差**：若审计判分器与开发判分器共享系统偏差（λ>0，如都偏爱"看起来合理的长输出"），held-out 也不够。λ 只能靠**与偏差无关的参照判分器**（可执行检查/专家面板/固定标注集）来锚定——这正是我们选"数学+符号验证"域的深层理由：可执行验证器是唯一能锚定 λ 的判分器。
3. **对角不可识别**：若改进者是求解器状态的确定函数，轨迹数据原则上无法区分"改进者变好了"还是"状态变好改了"。C4 的识别**必须**离开对角线：把早期与晚期改进者作用于同一起点、同预算、审计其后代（swap test）。evolving-vs-frozen 对照（=我们的 arm_W vs arm_WH）回答的是另一个问题，不能替代 swap test。

### 10.3 综述的操作化定义（判据供引用）
- 形式化：Π=(Γ,{ψ},{C},{𝒲})；自进化策略 f(Π,τ,r)=Π′；目标 max Σ U(Π_j,T_j)
- 三条准入：(i) 经验依赖且针对自身局限；(ii) 持久改变策略（非瞬时指令遵循）；(iii) 具备自主探索/自发起学习机制（active≠passive）
- 原则：「**不是由算法定义，而是由自主性的位置定义**」（locus of autonomy）
- 谱系：proto-evolution → strong self-evolution；与终身学习/课程学习/模型编辑的六维对照表（runtime evolving/dynamic/test-time/active/structural/self-reflect）

### 10.4 我们的主张声明（claim-indexed reporting，写进报告首页）
- **主张 C1+C2**：双通道交替在单机/LoRA 规模产生密封 held-out 上的产品增益且跨代累积（主实验+四变体归因）
- **主张 C4（消融）**：swap test——gen-0 改进者 (W_0,H_0) vs gen-4 改进者 (W_4,H_4) 作用于同一 fresh 基座、同预算训练一代、审计后代
- **不主张 C5**（5 代视野太短）；**C3** 用 arm_WH 第二 seed 完整重跑一次满足「≥2 reruns」
- 修正 arm 叙述：arm_W vs arm_WH = evolving-vs-frozen 对照，回答「让 harness 进化在整个 run 中是否有价值」，不是 C4

### 10.5 新增 P3 实验（由 10.2 直接导出，均便宜）
1. **空对照臂**：Phase A/B 的判分反馈随机打乱（选择照常进行）→ held-out 应平坦；其 dev 曲线涨幅 = 虚假增益的直接估计
2. **swap test**：见 10.4
3. **第二 seed 全环重跑**（C3 证据）
晋升门 margin δ 的理论依据 = 命题 1 的噪声感知接受规则。

---

## 11. 架构决策：独立构建 vs MCP 插件化（2026-10-31 定稿）

**决策：P0–P2 独立构建最小编排器，不做任何形式的插件化。**
理由：①三臂对照要求改进面外同构，通用 harness 是混杂变量源；②纪律§1（改进者无手）在工具丰富的 host 中不可执行；③插件化+frozen API 模型 = 权重通道不存在 = 重建 arm_H（恰好是本研究的批判对象）；④无论宿主是否参与决策，独立外环的代码都省不掉。
**MCP 的定位（第二阶段，demo 验证后）**：不是把 agent 插进别人的 harness，而是把实验室做成 MCP 服务（env_gen / grade / lockbox / archive / orchestrator 五个工具），让任何 harness 当"实验员"来使用——AAR 结构（agent+沙箱+评分服务器）。此时宿主的"有手"不违纪律：它在改进面之外。
**开发工具与实验对象的区分**：用 AI 辅助开发编排器代码不污染实验（工具在改进面外），此区分写进 README。
