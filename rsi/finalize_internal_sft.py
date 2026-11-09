"""finalize_internal_sft：内部自生成 SFT 跑完后，生成 v8 并推送 GitHub。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config


def main() -> None:
    summary_path = config.ROOT / "arms_sft/arm_W_internal_1500/summary.json"
    if not summary_path.exists():
        print("summary 不存在，结束")
        return
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    base = s["base"]
    sft = s["sft"]

    doc = f"""# test-RSI 实验记录 v8 —— 内部自生成 1,500 条 RSR-lite SFT

> 记录时间：{s.get('finished_at')}
> 训练数据：{s['train_n']} 条 RSR-lite 自生成中文题
> 目标：验证同域数据量是否影响 LoRA SFT
> 训练：LoRA iters={s['train_res'].get('iters')}, lr=5e-5

---

## 1. 结果

| 指标 | Base | SFT |
|---|---:|---:|
| 原 held-out 450 | {base['original_accuracy']*100:.2f}% | {sft['original_accuracy']*100:.2f}% |
| 外部中文 200 | {base['external_accuracy']*100:.2f}% | {sft['external_accuracy']*100:.2f}% |
| 混合 650 | {base['mixed_accuracy']*100:.2f}% | {sft['mixed_accuracy']*100:.2f}% |

## 2. 数据规模

- RSR-lite 尝试：{s.get('attempts')} 条
- 接受：{s.get('accepted')} 条
- train：{s.get('train_n')} 条
- valid：{s.get('valid_n')} 条

## 3. 结论

见最终分析。

## 4. 文件

- summary：`arms_sft/arm_W_internal_1500/summary.json`
- 日志：`logs/run_internal_sft.log`
"""
    (config.ROOT / "v8_internal_sft.md").write_text(doc, encoding="utf-8")

    subprocess.run(["git", "add", "v8_internal_sft.md",
                    "rsi/run_internal_sft.py", "rsi/finalize_internal_sft.py"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "v8 internal self-generated SFT result"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=config.ROOT, check=True)
    print("finalize done and pushed")


if __name__ == "__main__":
    main()
