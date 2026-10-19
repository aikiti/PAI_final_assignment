#!/usr/bin/env python3
"""Q-SELF-P 収録用: 出荷 policy_server.py の「起動時 seed」だけを環境変数で変えられる複製を作る純テキストパッチ。

正本: work/communications/FABLE_RUNCARDS_HONSEN2_FIRST72H_20260921.md「カード Q-SELF-P」noise seed の変え方。
  出荷サーバは起動時に定数 `PI05_PROCESS_SEED = 20260829` を 1 回張るだけで、`/reset` の seed は届かない
  （変更不可域のハンドラが渡さない）。同じ init を別のノイズ系列で走らせるには起動時 seed を変えるしかない。

このパッチがやること（それ以外は 1 バイトも変えない）:
  `PI05_PROCESS_SEED = 20260829`  →  `PI05_PROCESS_SEED = int(os.environ.get("QSELF_PROCESS_SEED", "20260829"))`
  ・環境変数が無ければ出荷物と同じ定数になる（= 挙動は出荷物と同一。既存ログとの決定論の照合に使う）。
  ・seed はプロセス起動時に 1 回張られるだけ。episode・task・受信 seed をキーにした分岐は入らない。
  ・起動時の `[pi05] process seed N` の 1 行（出荷物に元からある）で実効値が分かる。収録器はこれを読んで plan と照合する。
  ・**出荷物には入れない。** 複製は scratch の展開先に `policy_server_qself.py` として置き、zip は作らない。

変更不可領域（"# 以下は変更不可" のブロック以降）はバイト一致を検査する。
前例: work/tools/noise_patch_policy_server.py・te30_patch_policy_server.py と同じ差し込み方式。
使い方: python3 seed_patch_policy_server.py <in policy_server.py> <out policy_server_qself.py>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

BASE_SHA256 = "558fd4a409c359065f0089a8d4eb866b9ad4e32529ff8859d609e9dc4ca2f31f"  # T2F1 zip 内の policy_server.py
TAU05_SHA256 = "74160b40079dc78bdee8251a86c3be1e4e5662d516f0ca6d53ea63e0dce08962"  # 凍結 τ0.5 zip 内（full48k＋τ0.5 も同じ server）
# 10/7 LB の SMOOTH-X＋τ0.5（build_pi05_tau_overlay_candidate.py の出力。zip 内の server を SHA で凍結）
TAU05_OVERLAY_SHA256S = {
    "1f821de26644e8d21da54781597fa5dd7473b415b6ce6ab85dc9509be9aae6a9": "pi05_t2f1_smx1_tau05_nb_lic.zip",
    "984fd94503944a74d13c565b23e60f11a3e3171da52415cf13443d6c1a9e4761": "pi05_t2f1_smx3_tau05_nb_lic.zip",
}
MARKER = "# ============================================================\n# 以下は変更不可\n# ============================================================\n"
SEED_ANCHOR = "PI05_PROCESS_SEED = 20260829\n"
ENV_NAME = "QSELF_PROCESS_SEED"
SEED_PATCHED = (
    "# Q-SELF-P 収録用の複製（出荷物ではない）: 起動時 seed を環境変数で変えられる。未設定なら出荷物と同じ定数。\n"
    f'PI05_PROCESS_SEED = int(os.environ.get("{ENV_NAME}", "20260829"))\n'
)
SEED_USE = "        torch.manual_seed(PI05_PROCESS_SEED)\n"
SEED_PRINT = '        print(f"[pi05] process seed {PI05_PROCESS_SEED}", flush=True)\n'


def apply_seed_patch(src: str, allow_tau05: bool = False) -> str:
    """土台 policy_server.py のテキストへ差し込む。前提が崩れていれば AssertionError。"""
    assert src.count(MARKER) == 1, "変更不可マーカーが 1 回でない"
    head, tail = src.split(MARKER, 1)
    assert ENV_NAME not in src, "既に seed パッチ済み"
    if allow_tau05:
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
        assert sha == TAU05_SHA256 or sha in TAU05_OVERLAY_SHA256S, "τ0.5 土台の SHA が凍結値と違う"
        assert head.count("_T3_NOISE_TAU = 0.50\n") == 1, "τ0.5 の定義が想定外"
    for other in (("_CAG_OMEGA", "_te30_mix", "_T3_GAIN") if allow_tau05
                  else ("_T3_NOISE_TAU", "_CAG_OMEGA", "_te30_mix", "_T3_GAIN")):
        assert other not in src, f"他の推論側パッチ（{other}）が入った土台には当てない"
    assert head.count(SEED_ANCHOR) == 1, "PI05_PROCESS_SEED の定義が 1 回でない"
    assert head.count(SEED_USE) == 1, "起動時の torch.manual_seed(PI05_PROCESS_SEED) が 1 回でない"
    assert head.count(SEED_PRINT) == 1, "起動時の process seed の表示が 1 回でない"
    assert "\nimport os\n" in head[: head.index(SEED_ANCHOR)], "seed の定義より前に import os が無い"
    head = head.replace(SEED_ANCHOR, SEED_PATCHED)
    out = head + MARKER + tail
    assert out.split(MARKER, 1)[1] == tail, "変更不可領域が変わった"
    # 差分が seed の定義 1 箇所だけであることを行単位で確かめる。
    removed = [l for l in src.splitlines() if l not in out.splitlines()]
    added = [l for l in out.splitlines() if l not in src.splitlines()]
    assert removed == [SEED_ANCHOR.rstrip("\n")], f"想定外の削除: {removed}"
    assert added == SEED_PATCHED.rstrip("\n").split("\n"), f"想定外の追加: {added}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-tau05", action="store_true", help="SHA 凍結済みの τ0.5 版だけを許す")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args(argv)
    if a.dst.name == "policy_server.py":
        print("★ 出力名を policy_server.py にしない（出荷ファイルと取り違えないため）")
        return 1
    src = a.src.read_text(encoding="utf-8")
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    if sha != BASE_SHA256:
        print(f"※ 土台 sha256 {sha} は既知の T2F1 版 {BASE_SHA256} と違う（アンカー検査で最終判定する）")
    out = apply_seed_patch(src, allow_tau05=a.allow_tau05)
    a.dst.write_text(out, encoding="utf-8")
    print(f"SEED_PATCH_OK env={ENV_NAME} lines {len(src.splitlines())} -> {len(out.splitlines())} "
          f"immutable_tail_identical=True base_sha256={sha} "
          f"out_sha256={hashlib.sha256(out.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
