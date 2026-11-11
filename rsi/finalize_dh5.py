"""finalize_dh5：D×H₅ 评估完成后生成 v15 文档并推送 GitHub。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config


def main() -> None:
    summary_path = config.ROOT / "arms_sft/dh5/summary.json"
    if not summary_path.exists():
        print("summary 不存在，结束")
        return
    s = json.loads(summary_path.read_text(encoding="utf-8"))

    def pct(x):
        return f"{x*100:.2f}%"

    lines = [
        "# v15 —— D 权重 × H₅ harness 协同评估",
        "",
        f"> 完成时间：{s.get('finished_at')}",
        "> 测试集：原 held-out 450 + 外部中文 200 = 650 题",
        "> 目的：验证当前最佳权重 D 与当前最佳 harness H₅ 是否产生协同。",
        "",
        "## 结果",
        "",
        "| 变体 | 原 held-out 450 | 外部中文 200 | 混合 650 |",
        "|---|---:|---:|---:|",
        f"| Base + H₀ | {pct(s['base_h0']['original_accuracy'])} | {pct(s['base_h0']['external_accuracy'])} | {pct(s['base_h0']['mixed_accuracy'])} |",
        f"| D + H₀（w_only） | {pct(s['d_h0']['original_accuracy'])} | {pct(s['d_h0']['external_accuracy'])} | {pct(s['d_h0']['mixed_accuracy'])} |",
        f"| Base + H₅ core | {pct(s['h_core']['original_accuracy'])} | {pct(s['h_core']['external_accuracy'])} | {pct(s['h_core']['mixed_accuracy'])} |",
        f"| Base + H₅ full | {pct(s['h_full']['original_accuracy'])} | {pct(s['h_full']['external_accuracy'])} | {pct(s['h_full']['mixed_accuracy'])} |",
        f"| **D + H₅（sys）** | **{pct(s['sys']['original_accuracy'])}** | **{pct(s['sys']['external_accuracy'])}** | **{pct(s['sys']['mixed_accuracy'])}** |",
        "",
        "## 协同判定",
        "",
        f"- `sys − w_only`：{s['synergy']['sys_vs_w_only']*100:+.2f}pp",
        f"- `sys − h_full`：{s['synergy']['sys_vs_h_full']*100:+.2f}pp",
        f"- `sys > max(w_only, h_full)`：**{'是' if s['synergy']['sys_above_max_single'] else '否'}**",
        "",
        "## 结论",
        "",
    ]
    if s["synergy"]["sys_above_max_single"]:
        lines += [
            "- D 与 H₅ 存在正向协同；",
            "- `sys` 同时超过纯权重和纯 harness；",
            "- 可以进入一代真正 RSI 共进化实验。",
        ]
    else:
        lines += [
            "- D 与 H₅ 没有形成正向协同；",
            "- 当前最佳组合仍不超过单通道最优；",
            "- 不建议立即启动 W₆/H₆ 共进化，应先回滚到当前最佳单通道。",
        ]

    lines += [
        "",
        "## 分析",
        "",
        "1. 本轮统一在 650 题混合测试集上评估，避免不同题量带来的误比。",
        "2. `w_only` 复用 D + H₀ 的既有结果；`h_core/h_full/sys` 为本轮新增评估。",
        "3. 判断协同的标准是：`sys > max(w_only, h_full)`，且 parse_fail 不显著恶化。",
        "4. 若协同成立，下一步才启动 `(D,H₅) → H₆ → W₆` 的一代共进化。",
        "",
        "## 文件",
        "",
        "- 汇总：`arms_sft/dh5/summary.json`",
        "- 日志：`logs/eval_dh5.log`",
    ]
    (config.ROOT / "v15_dh5_synergy.md").write_text("\n".join(lines), encoding="utf-8")

    subprocess.run(["git", "add", "v15_dh5_synergy.md", "rsi/eval_dh5.py", "rsi/finalize_dh5.py"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "v15 D weights x H5 harness synergy result"], cwd=config.ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=config.ROOT, check=True)
    print("finalize_dh5 done and pushed")


if __name__ == "__main__":
    main()
