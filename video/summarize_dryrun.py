"""講座課題の録画で走らせた試運転の結果（最初の回）を、10/2 の M-DEMO-1 の記録と並べて表示する。表示だけで判定はしない。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORIG = Path("work/reports/honsen2_mdemo1_eval_20260926/runs")


def first_episodes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") != "episode" or int(r["episode_id"]) != 0:
            continue
        tag = "成功" if r["safe_success"] else ("衝突" if r["collided"] else "失敗")
        out[r["group"]] = f"{tag} {r['steps']} step"
    return out


def main() -> int:
    dry = Path(sys.argv[1])
    for f in sorted((dry / "runs").glob("mdemo1_*_main_kit_s*.jsonl")):
        now = first_episodes(f)
        orig_path = ORIG / f.name
        before = first_episodes(orig_path) if orig_path.exists() else {}
        for group, val in now.items():
            print(f"[録画の回] {f.stem} {group}: 今回 {val} ／ 10/2 の記録 {before.get(group, '記録なし')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
