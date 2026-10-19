#!/usr/bin/env python3
"""凍結済みの G-ZERO マスクに「判定に効くコードの来歴」の別紙（sidecar）を付ける。マスク本体は変更しない。

Codex 2026-09-26 裁定 P2「旧マスクを改変せず、来歴の補足を別版/sidecar で記録」。
マスクを作った G-ZERO の走行（run_meta）には判定コードの来歴（`reset_contract.runtime_contract`）が入っていなかった。
いまの来歴がその走行のときと同じだと言える根拠を集めてから、sidecar に書く:
  ・rollout.py・environment.py の sha が、走行の run_meta に記録された sha と一致する
  ・判定に効く他のファイル（採点の既定値・設定・接触ガード）の更新時刻が、走行の開始より前
  ・LIBERO-plus の git 管理下の *.py と改変ファイルの更新時刻が、走行の開始より前
1 つでも満たさなければ書かない（来歴を言い切れない）。
使い方: python3 work/eval/b3_panel/mask_contract_sidecar.py --mask <mask.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import reset_contract as rc  # noqa: E402


def sidecar_path(mask: Path) -> Path:
    """v2（runtime-contract/v2: LIBERO はコードの中身だけ）。v1 の `.contract_sidecar.json` は .DS_Store を含む版で、使わない（残す）。"""
    return mask.with_name(mask.stem + ".contract_v2.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mask", type=Path, required=True)
    a = ap.parse_args()
    a.mask = a.mask.resolve()
    mask = json.loads(a.mask.read_text())
    run_path = REPO / mask["gzero_run"]["path"]
    if rc.sha256_bytes(run_path.read_bytes()) != mask["gzero_run"]["sha256"]:
        raise SystemExit("★ G-ZERO の走行ファイルがマスクの記録と違う")
    meta = json.loads(run_path.read_text().splitlines()[0])
    started = dt.datetime.fromisoformat(meta["started_at"]).timestamp()
    if isinstance(meta.get("runtime_contract"), dict):
        # 来歴の補強（P2）の後の走行: run_meta に記録した来歴をそのまま使う（いまのファイルとの一致は候補側の照合で見る）
        write_sidecar(a.mask, mask, meta, meta["runtime_contract"], {"source": "G-ZERO の run_meta に記録された runtime_contract"})
        return 0
    contract = rc.runtime_contract(REPO)
    problems, evidence = [], {}
    for key, path in (("rollout_sha256", "PARC2026_final/pipeline/rollout.py"), ("environment_sha256", "PARC2026_final/pipeline/environment.py")):
        ok = contract["files"][path] == meta["runtime"][key]
        evidence[path] = {"sha_matches_run_meta": ok}
        if not ok:
            problems.append(f"{path} の sha が走行時と違う")
    for p in rc.CONTRACT_FILES:
        mt = (REPO / p).stat().st_mtime
        evidence.setdefault(p, {})["mtime"] = dt.datetime.fromtimestamp(mt).astimezone().isoformat(timespec="seconds")
        if mt >= started:
            problems.append(f"{p} が走行の開始より後に更新されている")
    lib = REPO / "PARC2026_pre-main/LIBERO-plus"
    listed = subprocess.run(["git", "-C", str(lib), "ls-files", "*.py"], capture_output=True, text=True, check=True).stdout.split()
    changed = subprocess.run(["git", "-C", str(lib), "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=True).stdout.split()
    newest = max((lib / f).stat().st_mtime for f in listed + changed if (lib / f).exists())
    evidence["libero_plus"] = {"n_listed_py": len(listed), "changed_files": changed,
                               "newest_mtime": dt.datetime.fromtimestamp(newest).astimezone().isoformat(timespec="seconds")}
    if newest >= started:
        problems.append("LIBERO-plus のファイルが走行の開始より後に更新されている")
    if problems:
        raise SystemExit("★ 来歴を言い切れないので sidecar を書かない: " + "; ".join(problems))
    write_sidecar(a.mask, mask, meta, contract, evidence)
    return 0


def write_sidecar(mask_path: Path, mask: dict, meta: dict, contract: dict, evidence: dict) -> None:
    body = {"kind": "gzero_mask_contract_sidecar", "schema": 1,
            "note": ("マスク本体は変更していない。マスクを作った走行の run_meta に記録された、判定に効くコードの来歴。"
                     if evidence.get("source") else
                     "マスク本体は変更していない。マスクを作った走行の後に、判定に効くコードが変わっていないことを確かめたうえでの来歴の補足。"),
            "mask": {"path": str(mask_path.resolve().relative_to(REPO)), "sha256": rc.sha256_bytes(mask_path.read_bytes())},
            "gzero_run": {**mask["gzero_run"], "started_at": meta["started_at"]},
            "contract": contract, "contract_fingerprint": rc.contract_fingerprint(contract), "evidence": evidence,
            "ruling": "work/communications/codex/CODEX_RULING_B3_CONFIRM_MANUAL_20260926.md P2"}
    out = sidecar_path(mask_path)
    if out.exists():
        old = json.loads(out.read_text())
        if {k: v for k, v in old.items() if k != "created_at"} != body:
            bad = sorted(k for k in set(old) | set(body) if k != "created_at" and old.get(k) != body.get(k))
            raise SystemExit(f"★ 既存の sidecar {out} と中身が違う（{bad}）。上書きしない")
        print(f"既存と同一: {out.relative_to(REPO)}")
    else:
        out.write_text(json.dumps({**body, "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")},
                                  ensure_ascii=False, indent=1) + "\n")
        print(f"作成: {out.relative_to(REPO)}")
    print(f"contract_fingerprint {body['contract_fingerprint']}  sha256 {rc.sha256_bytes(out.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
