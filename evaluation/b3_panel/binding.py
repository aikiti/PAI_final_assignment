#!/usr/bin/env python3
"""評価ラベルと export の結び付け（Codex 2026-09-26 裁定 P2）。B3・B4 の評価台本が、再開・SKIP・作業 root の再利用の前に照合する。

結び付ける物: ラベル → export の内容 hash（`gpu_smoke_test.content_hash`。パスの文字列ではなく中身）→ 前処理・後処理・config・
tokenizer の各 hash → 評価に使う server の実装（seed を差し込んだ複製の sha）→ 起動時 seed。
受領物を同じ名前のパスへ置き換えた・ラベルに別の export を渡した、を検出する。

  make  --label L --export DIR --server-sha SHA --seed N --out FILE
        結び付けを作る。FILE が既にあれば中身を照合し、違えば終了コード 3（上書きしない）。同じなら 0。
  check --binding FILE --export DIR [--server-sha SHA --seed N]
        いまの export（と server・seed）が結び付けと同じか。違えば終了コード 3。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "work/tools"))

PROCESSOR_FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(export: Path) -> dict:
    from gpu_smoke_test import content_hash
    export = Path(export)
    tok = export / "tokenizer"
    return {"export_content_hash": content_hash(export),
            "processor_sha256": {f: (sha_file(export / f) if (export / f).is_file() else None) for f in PROCESSOR_FILES},
            "tokenizer_content_hash": (content_hash(tok) if tok.is_dir() else None),
            "weights_sha256": (sha_file(export / "model.safetensors") if (export / "model.safetensors").is_file() else None)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    mk = sp.add_parser("make")
    mk.add_argument("--label", required=True)
    mk.add_argument("--export", type=Path, required=True)
    mk.add_argument("--server-sha", required=True)
    mk.add_argument("--seed", type=int, required=True)
    mk.add_argument("--out", type=Path, required=True)
    ck = sp.add_parser("check")
    ck.add_argument("--binding", type=Path, required=True)
    ck.add_argument("--export", type=Path, required=True)
    ck.add_argument("--server-sha", default=None)
    ck.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        body = {"kind": "eval_binding", "schema": 1, "label": a.label, "export_path_at_bind": str(a.export),
                **describe(a.export), "server_sha256": a.server_sha, "process_seed": a.seed}
        if a.out.exists():
            old = json.loads(a.out.read_text())
            keys = [k for k in body if k != "export_path_at_bind"]
            bad = [k for k in keys if old.get(k) != body[k]]
            if bad:
                print(f"★ ラベル {a.label} の結び付けと違う: {bad}（上書きしない。別の export なら別のラベルで）", file=sys.stderr)
                return 3
            print(f"結び付けと一致: {a.out}")
            return 0
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({**body, "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")},
                                    ensure_ascii=False, indent=1) + "\n")
        print(f"結び付けを作成: {a.out}  export {body['export_content_hash'][:16]}")
        return 0
    old = json.loads(a.binding.read_text())
    now = describe(a.export)
    bad = [k for k in now if old.get(k) != now[k]]
    if a.server_sha is not None and old.get("server_sha256") != a.server_sha:
        bad.append("server_sha256")
    if a.seed is not None and old.get("process_seed") != a.seed:
        bad.append("process_seed")
    if bad:
        print(f"★ export・server・seed が結び付け {a.binding} と違う: {bad}", file=sys.stderr)
        return 3
    print(f"結び付けと一致: {a.binding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
