"""排除账本：一源五角色一账本的「账本」侧。

角色：
  heldout / calib      → 全局账本（_global.json），所有臂共享排除
  probe / pool / promo-val → 分臂账本（{arm}.json），三臂是三个平行世界

账本记录每个题实例 ID 的角色与元信息；env_gen 出题前查询账本，
保证任何实例 ID 全局唯一、绝不跨角色复用（防泄漏的第三层防线）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import config


def _ledger_path(arm: str) -> Path:
    if arm == "GLOBAL":
        return config.GLOBAL_LEDGER
    return config.LEDGER_DIR / f"{arm}.json"


class Ledger:
    """单次出题会话持有的账本视图（global + 指定 arm 两册合并）。"""

    def __init__(self, arm: str = "GLOBAL"):
        self.arm = arm
        self.path_global = config.GLOBAL_LEDGER
        self.path_arm = _ledger_path(arm)
        self.entries: dict[str, dict] = {}

        for p in (self.path_global, self.path_arm):
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self.entries.update(data)

    def __contains__(self, pid: str) -> bool:
        return pid in self.entries

    def used_ids(self) -> set[str]:
        return set(self.entries)

    def assign(self, pid: str, role: str, meta: dict | None = None) -> None:
        if role not in config.ROLES:
            raise ValueError(f"未知角色: {role}")
        if pid in self.entries:
            raise ValueError(f"实例 {pid} 已分配为 {self.entries[pid]['role']}，禁止复用")
        rec = {"role": role, "arm": self.arm, "ts": time.time()}
        if meta:
            rec.update(meta)
        self.entries[pid] = rec

    def save(self) -> None:
        """把新条目按角色拆回两册账本，原子写盘。"""
        config.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        for p in (self.path_global, self.path_arm):
            merged: dict[str, dict] = {}
            if p.exists():
                merged = json.loads(p.read_text(encoding="utf-8"))
            for pid, rec in self.entries.items():
                is_global_role = rec["role"] in config.GLOBAL_ROLES
                belongs_here = (p == self.path_global) == is_global_role
                if belongs_here:
                    if pid in merged and merged[pid] != rec:
                        raise RuntimeError(f"账本冲突: {pid}")
                    merged[pid] = rec
            if merged:
                tmp = p.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                os.replace(tmp, p)


def stats() -> dict:
    """全部账本的角色统计（status.py / 报告用）。"""
    out: dict[str, dict[str, int]] = {}
    for p in sorted(config.LEDGER_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        per_role: dict[str, int] = {}
        for rec in data.values():
            per_role[rec["role"]] = per_role.get(rec["role"], 0) + 1
        out[p.stem] = per_role
    return out
