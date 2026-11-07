"""taskbank：固定题库的生成、校验与读取。

用途
----
把实验要用的学术题一次性生成好，按角色和世代切成互不重叠的固定分片：

    data/frozen_academic_v1/
    ├── manifest.json
    ├── train/pool_gen_001.jsonl ...
    ├── probe/probe_gen_001.jsonl ...
    └── promo/promo_gen_001.jsonl ...

设计纪律
--------
- 题库只生成一次，实验过程中禁止再临时出题；
- 题目 ID 全局唯一，不同角色/分片绝不重叠；
- 每个分片和 held-out 都有 SHA256，读取前校验；
- 三臂共享同一份题目清单，但各臂的模型轨迹和 adapter 分开；
- held-out 仍只由 lockbox 读取，不复制进题库目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from . import config, env_gen

BANK_NAME = "frozen_academic_v1"
DEFAULT_BANK_ROOT = config.ROOT / "data" / BANK_NAME
DEFAULT_GENS = 5
DEFAULT_TRAIN_PER_GEN = 100
DEFAULT_PROBE_PER_GEN = 50
DEFAULT_PROMO_PER_GEN = 50
DEFAULT_DIFFICULTY = 5
DEFAULT_SEED = 20261101


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _canonical_hash(obj: dict) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 一次性生成
# ---------------------------------------------------------------------------

def _generate_split(
    n: int,
    difficulty: int,
    rng: random.Random,
    used_ids: set[str],
    template_names: list[str],
) -> list[dict]:
    """生成一个固定分片，保证分片内和全局 ID 唯一。

    模板按轮转方式使用，避免某个模板把分片占满。
    """
    if n <= 0:
        return []
    if not template_names:
        raise ValueError("template_names 为空")

    order = list(template_names)
    rng.shuffle(order)
    out: list[dict] = []
    attempts = 0
    max_attempts = n * 200 + 2000
    fail_streak: dict[str, int] = {t: 0 for t in order}

    while len(out) < n and attempts < max_attempts:
        tname = order[attempts % len(order)]
        attempts += 1
        try:
            params, question, answer = env_gen.TEMPLATES[tname](rng, int(difficulty))
        except RuntimeError:
            fail_streak[tname] += 1
            # 某个模板参数空间耗尽时，暂时从轮转中移除。
            if fail_streak[tname] > 1000 and len(order) > 1:
                order.remove(tname)
                fail_streak.pop(tname, None)
            continue

        pid = env_gen.make_id(tname, params)
        if pid in used_ids:
            fail_streak[tname] += 1
            if fail_streak[tname] > 500 and tname in order and len(order) > 1:
                order.remove(tname)
                fail_streak.pop(tname, None)
            continue

        used_ids.add(pid)
        fail_streak[tname] = 0
        out.append(
            {
                "id": pid,
                "template": tname,
                "difficulty": int(difficulty),
                "question": question,
                "answer": env_gen.frac_str(answer),
                "answer_frac": str(answer),
                "params": params,
            }
        )

    if len(out) != n:
        raise RuntimeError(
            f"固定题库生成不足：需要 {n}，只生成 {len(out)}；"
            "请扩大模板参数空间或降低每个分片的题量。"
        )
    return out


def generate_bank(
    out_dir: Path | str = DEFAULT_BANK_ROOT,
    gens: int = DEFAULT_GENS,
    train_per_gen: int = DEFAULT_TRAIN_PER_GEN,
    probe_per_gen: int = DEFAULT_PROBE_PER_GEN,
    promo_per_gen: int = DEFAULT_PROMO_PER_GEN,
    difficulty: int = DEFAULT_DIFFICULTY,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict:
    """一次性生成并冻结题库，返回 manifest。

    该函数只负责题目；模型生成的解答轨迹不在题库里。
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if out_dir.exists() and not force:
        raise SystemExit(f"题库目录已存在：{out_dir}。如需重建请显式加 --force。")
    if not (1 <= int(difficulty) <= 5):
        raise ValueError("difficulty 必须在 1-5")

    # 保留 held-out 原文件不动，只把它的路径和哈希写入 manifest。
    heldout_path = config.HELDOUT_FILE
    if not heldout_path.exists():
        raise SystemExit(f"held-out 文件不存在：{heldout_path}；先运行 lockbox/freeze.py")
    heldout_sha = sha256_of(heldout_path)
    expected_sha_file = config.HELDOUT_SHA
    if expected_sha_file.exists():
        expected = expected_sha_file.read_text(encoding="utf-8").split()[0].strip()
        if expected != heldout_sha:
            raise SystemExit("held-out SHA256 校验失败，拒绝生成题库。")

    # held-out ID 从一开始就保留，避免生成过程中撞题。
    heldout_recs = _read_jsonl(heldout_path)
    heldout_ids = {r["id"] for r in heldout_recs}
    out_dir.mkdir(parents=True, exist_ok=True)
    template_names = list(env_gen.TEMPLATE_NAMES)
    used_ids: set[str] = set(heldout_ids)
    splits: dict[str, dict] = {}

    for g in range(1, int(gens) + 1):
        for role, n in (
            ("train", int(train_per_gen)),
            ("probe", int(probe_per_gen)),
            ("promo", int(promo_per_gen)),
        ):
            rng = random.Random(f"taskbank:{seed}:{difficulty}:{role}:{g}")
            recs = _generate_split(n, int(difficulty), rng, used_ids, template_names)
            for r in recs:
                r["role"] = role
                r["gen"] = g
                r["bank"] = BANK_NAME
            rel = Path(role) / f"{role}_gen_{g:03d}.jsonl"
            path = out_dir / rel
            _write_jsonl(path, recs)
            splits[f"{role}/gen_{g:03d}"] = {
                "role": role,
                "gen": g,
                "path": str(rel),
                "n": len(recs),
                "sha256": sha256_of(path),
            }

    # 防止 held-out 和题库 ID 重叠；held-out 本身不复制。
    generated_ids = used_ids - heldout_ids
    collision = sorted(generated_ids & heldout_ids)
    if collision:
        raise SystemExit(f"题库与 held-out 出现 {len(collision)} 个重复 ID，拒绝冻结。")

    manifest = {
        "schema_version": 1,
        "bank": BANK_NAME,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "master_seed": int(seed),
        "difficulty": int(difficulty),
        "gens": int(gens),
        "counts": {
            "train_per_gen": int(train_per_gen),
            "probe_per_gen": int(probe_per_gen),
            "promo_per_gen": int(promo_per_gen),
            "total_generated": sum(v["n"] for v in splits.values()),
        },
        "splits": splits,
        "heldout": {
            "path": str(heldout_path.relative_to(config.ROOT)),
            "n": len(heldout_ids),
            "sha256": heldout_sha,
        },
        "sharing_rule": "三臂共享题目清单；轨迹、训练数据和 adapter 按臂独立。",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# 读取与校验
# ---------------------------------------------------------------------------

def _manifest_path(root: Path | str | None = None) -> Path:
    root = Path(root) if root else DEFAULT_BANK_ROOT
    return root / "manifest.json"


def load_manifest(root: Path | str | None = None, verify: bool = True) -> dict:
    path = _manifest_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"固定题库不存在：{path}\n"
            "先运行：python -m rsi.taskbank generate --gens 5"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if verify:
        verify_bank(root, manifest=manifest)
    return manifest


def verify_bank(root: Path | str | None = None, manifest: dict | None = None) -> dict:
    """校验 manifest、所有分片哈希以及跨分片/held-out 唯一性。"""
    root = Path(root) if root else DEFAULT_BANK_ROOT
    manifest = manifest or json.loads(_manifest_path(root).read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_sha256", None)
    actual_manifest_hash = _canonical_hash(manifest)
    manifest["manifest_sha256"] = expected_manifest_hash
    if expected_manifest_hash != actual_manifest_hash:
        raise RuntimeError("题库 manifest 校验失败：manifest 被改动过。")

    all_ids: set[str] = set()
    for key, info in manifest["splits"].items():
        path = root / info["path"]
        if not path.exists():
            raise FileNotFoundError(f"题库分片缺失：{path}")
        actual = sha256_of(path)
        if actual != info["sha256"]:
            raise RuntimeError(f"题库分片 SHA256 不符：{key}")
        recs = _read_jsonl(path)
        if len(recs) != info["n"]:
            raise RuntimeError(f"题库分片数量不符：{key}")
        ids = {r["id"] for r in recs}
        if len(ids) != len(recs):
            raise RuntimeError(f"题库分片内部有重复 ID：{key}")
        overlap = all_ids & ids
        if overlap:
            raise RuntimeError(f"跨分片重复 ID：{key}，示例 {sorted(overlap)[:3]}")
        all_ids |= ids

    heldout_path = root.parent.parent / manifest["heldout"]["path"]
    if not heldout_path.exists():
        heldout_path = config.ROOT / manifest["heldout"]["path"]
    if not heldout_path.exists():
        raise FileNotFoundError(f"held-out 文件缺失：{heldout_path}")
    if sha256_of(heldout_path) != manifest["heldout"]["sha256"]:
        raise RuntimeError("held-out SHA256 与题库 manifest 不一致")
    heldout_ids = {r["id"] for r in _read_jsonl(heldout_path)}
    overlap = all_ids & heldout_ids
    if overlap:
        raise RuntimeError(f"题库与 held-out 重复 ID：{sorted(overlap)[:3]}")
    return {"ok": True, "n_splits": len(manifest["splits"]),
            "n_generated": len(all_ids), "n_heldout": len(heldout_ids)}


def load_split(
    role: str,
    gen: int,
    root: Path | str | None = None,
    verify: bool = True,
    expected_n: int | None = None,
) -> list[dict]:
    """读取指定角色、指定世代的固定分片。role ∈ train/probe/promo。"""
    if role not in ("train", "probe", "promo"):
        raise ValueError(f"未知题库角色：{role}")
    manifest = load_manifest(root, verify=verify)
    key = f"{role}/gen_{int(gen):03d}"
    if key not in manifest["splits"]:
        raise KeyError(f"题库中没有分片：{key}")
    path = (Path(root) if root else DEFAULT_BANK_ROOT) / manifest["splits"][key]["path"]
    recs = _read_jsonl(path)
    if expected_n is not None and len(recs) < expected_n:
        raise RuntimeError(f"分片 {key} 只有 {len(recs)} 题，少于需要 {expected_n}")
    return recs


def split_meta(role: str, gen: int, root: Path | str | None = None) -> dict:
    manifest = load_manifest(root, verify=False)
    key = f"{role}/gen_{int(gen):03d}"
    return dict(manifest["splits"][key])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="生成/校验固定学术题库")
    sub = ap.add_subparsers(dest="cmd", required=True)

    gp = sub.add_parser("generate", help="一次性生成题库")
    gp.add_argument("--out-dir", type=Path, default=DEFAULT_BANK_ROOT)
    gp.add_argument("--gens", type=int, default=DEFAULT_GENS)
    gp.add_argument("--train-per-gen", type=int, default=DEFAULT_TRAIN_PER_GEN)
    gp.add_argument("--probe-per-gen", type=int, default=DEFAULT_PROBE_PER_GEN)
    gp.add_argument("--promo-per-gen", type=int, default=DEFAULT_PROMO_PER_GEN)
    gp.add_argument("--difficulty", type=int, default=DEFAULT_DIFFICULTY)
    gp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gp.add_argument("--force", action="store_true")

    vp = sub.add_parser("verify", help="校验题库")
    vp.add_argument("--root", type=Path, default=DEFAULT_BANK_ROOT)

    args = ap.parse_args()
    if args.cmd == "generate":
        manifest = generate_bank(
            out_dir=args.out_dir,
            gens=args.gens,
            train_per_gen=args.train_per_gen,
            probe_per_gen=args.probe_per_gen,
            promo_per_gen=args.promo_per_gen,
            difficulty=args.difficulty,
            seed=args.seed,
            force=args.force,
        )
        print(json.dumps({
            "bank": manifest["bank"],
            "counts": manifest["counts"],
            "splits": len(manifest["splits"]),
            "manifest_sha256": manifest["manifest_sha256"],
            "out_dir": str(args.out_dir),
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(verify_bank(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
