#!/usr/bin/env python3
"""B3/T3 評価の reset 契約の共通部品（Codex 2026-09-26 裁定 `work/communications/codex/CODEX_RULING_RESET_GZERO_20260926.md`）。

用語（裁定 (a)）:
  ③ kit   … 手元の配布 kit（PARC2026_final/pipeline/rollout.py）と同じ手順。task ごとに env を 1 個作って env.seed(env_seed)、
             同じ task の init を凍結した順に連続 reset する。**主評価**。「非公開の本番と同一」ではなく「手元の配布 kit 準拠」と呼ぶ。
  ① fresh … episode（init）ごとに env を作り直し、毎回 env.seed(env_seed)。**感度分析**。③ と数値を混ぜない。
  ② 家具の復元 … 収録時の配置へ家具を戻す。教師の再構成・原因診断だけ（このモジュールでは行わない。主評価で使わない）。

ここに置くもの:
  ・終端の分類（raw done がある新しい記録用と、raw done が無い既存 selfeval 記録用）
  ・env から指紋を取る関数。**乱数を消費しない**（np.random / random の状態を読むだけで書き換えない。家具はモデルを
    毎回取り直して読む。LIBERO は hard reset のたびに sim を作り直すので古い参照は使わない）。
  ・対応（G-ZERO と候補が同じ場面を見たか）の比較とキー。

公式スコアは計算しない。LB を予測しない。
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
from typing import Any, Iterable

import numpy as np

# ----------------------------------------------------------------------------- 終端の分類

SUCCESS = "success"                      # done かつ衝突なし（300 step 目の成功を含む）
DONE_COLLIDED = "done_collided"          # done だが非対象物が閾値を超えた
TIMEOUT = "timeout"                      # done に届かないまま上限に達した（= not done and reached_limit）
LIMIT_DONE_UNKNOWN = "limit_done_unknown"  # 既存記録だけ: 上限に達し衝突、raw done が無く done か未達か分からない
ANOMALY = "anomaly_not_done_before_limit"  # done でも上限でもないのに止まった（配布 kit の loop では起きないはず）
TERMINATIONS = (SUCCESS, DONE_COLLIDED, TIMEOUT, LIMIT_DONE_UNKNOWN, ANOMALY)

TERMINATION_JA = {SUCCESS: "成功", DONE_COLLIDED: "done 後の衝突", TIMEOUT: "時間切れ（done に届かず上限）",
                  LIMIT_DONE_UNKNOWN: "上限到達・done 不明（衝突あり）", ANOMALY: "異常（上限前に done なしで停止）"}


def classify_termination(done: bool, collided: bool, steps: int, max_steps: int) -> str:
    """raw done がある記録用。timeout = not done and reached_limit（Codex 裁定 (b)）。"""
    if done:
        return DONE_COLLIDED if collided else SUCCESS
    if int(steps) >= int(max_steps):
        return TIMEOUT
    return ANOMALY


def classify_legacy(success: bool, collided: bool, steps: int, max_steps: int) -> str:
    """raw done が無い既存 selfeval 記録用。done は success と step 数から確定できる範囲だけ使い、推測で埋めない。

    ・success → done かつ衝突なし（300 step 目の成功も成功）
    ・上限前に止まった失敗 → loop が止まるのは done のときだけなので done。衝突なら done 後の衝突
    ・上限で衝突なしの失敗 → done なら成功になるはずなので done ではない → 時間切れ
    ・上限で衝突ありの失敗 → done で衝突か、未達で衝突か区別できない → 上限到達・done 不明（timeout へ寄せない）
    """
    if success:
        return SUCCESS
    if int(steps) < int(max_steps):
        return DONE_COLLIDED if collided else ANOMALY
    return LIMIT_DONE_UNKNOWN if collided else TIMEOUT


def termination_counts(labels: Iterable[str]) -> dict[str, int]:
    labels = list(labels)
    return {t: sum(1 for x in labels if x == t) for t in TERMINATIONS}


# ----------------------------------------------------------------------------- 指紋（乱数を消費しない）

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_sha256(a: Any) -> str:
    """float64 に揃えた生バイトの sha256（init の行・状態ベクトルの同一性）。"""
    return sha256_bytes(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())


def rng_digest() -> dict[str, Any]:
    """NumPy のグローバル乱数（LIBERO の env.seed と家具の配置が使う）と Python の random の状態の digest。読むだけ。"""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    h = hashlib.sha256()
    h.update(str(kind).encode())
    h.update(np.ascontiguousarray(np.asarray(keys, dtype=np.uint32)).tobytes())
    h.update(struct.pack("<iid", int(pos), int(has_gauss), float(cached)))
    py = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()[:16]
    return {"np": h.hexdigest()[:16], "np_pos": int(pos), "py": py}


def _inner(env: Any) -> Any:
    """OffScreenRenderEnv（ControlEnv）→ LIBERO の問題 env。"""
    return getattr(env, "env", env)


def fixture_poses(env: Any) -> dict[str, dict[str, Any]]:
    """bddl の (:fixtures …) に定義された家具の root body の位置・姿勢（model.body_pos/quat）。

    Codex 裁定 (a)3: 家具は fixture 定義から特定し、「自由関節が無い body」一般を家具とみなさない。
    LIBERO の `_reset_internal` が reset のたびに書き換えるのは `fixtures_dict` の各 root body の body_pos/quat だけ
    （`bddl_base_domain.py` の _reset_internal）。モデルは毎回取り直す（hard reset で作り直されるため）。
    """
    inner = _inner(env)
    model = inner.sim.model
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(getattr(inner, "fixtures_dict", {}) or {}):
        obj = inner.fixtures_dict[name]
        body = obj.root_body
        bid = int(model.body_name2id(body))
        out[name] = {"body": body, "pos": [float(x) for x in model.body_pos[bid]],
                     "quat": [float(x) for x in model.body_quat[bid]]}
    return out


def fixtures_digest(poses: dict[str, dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for name in sorted(poses):
        h.update(name.encode())
        h.update(np.asarray(poses[name]["pos"] + poses[name]["quat"], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def state_fingerprint(sim: Any) -> dict[str, Any]:
    """sim の状態（time・qpos・qvel と flatten 全体）の指紋。"""
    st = sim.get_state()
    flat = np.asarray(st.flatten(), dtype=np.float64)
    return {"flat": array_sha256(flat)[:16], "qpos": array_sha256(st.qpos)[:16], "qvel": array_sha256(st.qvel)[:16],
            "time": float(st.time), "dim": int(flat.shape[0])}


# ----------------------------------------------------------------------------- キーと対応の検査

#: episode を突き合わせるキー（Codex 裁定 (b)）。推論系列の seed はキーに入れず別の列に持つ。
KEY_FIELDS = ("manifest_sha256", "task_stem", "bddl_sha256", "init_index", "init_state_sha256", "env_seed", "reset_mode",
              "episode_id", "reset_ordinal_in_env")

#: 同じキーの G-ZERO と候補で一致しなければならない場面の指紋（ずれたら昇格判定に使わない）。
#: 乱数は NumPy のグローバル乱数（LIBERO の env.seed と家具の配置が使う物）だけを比べる。Python の `random` はプロセス起動時に
#: OS の乱数で初期化されるので、同じ条件の 2 走行でも一致しない（2026-09-26 の小確認で確認。家具の配置はそれでも完全一致）。
CORRESPONDENCE_FIELDS = ("reset.rng_pre.np", "reset.rng_post.np", "reset.attempts", "reset.fixtures_digest",
                         "init.observed_sha256", "settle.state.flat", "settle.rng", "settle.steps")

#: 同じ比較に入れる走行の実行環境（違えば比較しない）。
RUNTIME_FIELDS = ("python", "numpy", "mujoco", "robosuite", "libero_plus_rev", "rollout_sha256", "environment_sha256",
                  "collision_threshold_m", "max_steps", "camera", "control_freq")


def dig(rec: dict[str, Any], dotted: str) -> Any:
    cur: Any = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fingerprint_get(fp: dict[str, Any], field: str) -> Any:
    """凍結済みマスクの fingerprint（項目名 → 値の平らな辞書）から値を引く。項目名の一部が辞書で保存されていれば中へ辿る
    （例: 旧い項目名 "reset.rng_pre" に {"np": …, "py": …} が入っている場合、"reset.rng_pre.np" はその "np"）。"""
    if field in fp:
        return fp[field]
    parts = field.split(".")
    for i in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:i])
        if head in fp:
            return dig(fp[head], ".".join(parts[i:])) if isinstance(fp[head], dict) else None
    return None


def episode_key(rec: dict[str, Any]) -> tuple:
    return tuple(rec.get(f) for f in KEY_FIELDS)


def compare_correspondence(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """同じキーの 2 つの episode 記録で、場面の指紋が食い違った項目を返す（空なら対応している）。

    float の許容誤差は置かない（同じ runtime なら完全一致が原則。Codex 裁定 (a)4。許容が要るなら候補を見る前に固定する）。
    """
    bad = []
    for f in CORRESPONDENCE_FIELDS:
        va, vb = dig(a, f), dig(b, f)
        if va is None or vb is None or va != vb:
            bad.append(f)
    return bad


def runtime_fingerprint(meta: dict[str, Any]) -> str:
    rt = meta.get("runtime", {})
    return sha256_bytes(json.dumps({k: rt.get(k) for k in RUNTIME_FIELDS}, sort_keys=True).encode())[:16]


# ----------------------------------------------------------------------------- 判定に効くコードの来歴（Codex 2026-09-26 P2）

#: 測定の中身（episode ループ・静定・成功と衝突の判定・採点の既定値・T1/T2 と T3 の接触ガード）を決める配布 kit のファイル。
CONTRACT_FILES = ("PARC2026_final/pipeline/rollout.py", "PARC2026_final/pipeline/environment.py",
                  "PARC2026_final/pipeline/total_score.py", "PARC2026_final/pipeline/config.py",
                  "PARC2026_final/compe/t1/register.py", "PARC2026_final/compe/t3/register.py")
#: LIBERO-plus の来歴。**Python のコードの中身だけ**で取る: git 管理下の *.py の内容（作業ツリーの改変を含む）と、
#: 改変された *.py の一覧と中身。git の dirty・patch・status は Finder の .DS_Store や __pycache__ で揺れるので入れない
#: （2026-09-26、LIBERO-plus の改変一覧に .DS_Store が 3 つ入っていた）。
LIBERO_CONTRACT_KEYS = ("rev", "source_tree_sha256", "changed_py")
CONTRACT_SCHEMA = "runtime-contract/v2"


def runtime_contract(repo: Any) -> dict[str, Any]:
    """判定に効くファイルの sha256 と LIBERO-plus のコードの来歴（`work/eval/selfeval/provenance.py` を再利用）。"""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(repo)
    files = {p: (sha256_bytes((repo / p).read_bytes()) if (repo / p).is_file() else "(missing)") for p in CONTRACT_FILES}
    if str(repo / "work/eval/selfeval") not in sys.path:
        sys.path.insert(0, str(repo / "work/eval/selfeval"))
    import provenance as pv
    lib = repo / "PARC2026_pre-main/LIBERO-plus"
    lp = pv.libero_plus_provenance(lib)
    names = subprocess.run(["git", "-C", str(lib), "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=False).stdout.split()
    changed_py = {p: (sha256_bytes((lib / p).read_bytes()) if (lib / p).is_file() else "(deleted)") for p in sorted(names) if p.endswith(".py")}
    return {"schema": CONTRACT_SCHEMA, "files": files,
            "libero_plus": {"rev": lp.get("rev"), "source_tree_sha256": lp.get("source_tree_sha256"), "changed_py": changed_py}}


def contract_fingerprint(contract: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(contract, sort_keys=True).encode())[:16]


def contract_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """2 つの runtime_contract で違う項目（ファイルのパス・LIBERO の項目名）。"""
    out = [p for p in sorted(set(a.get("files", {})) | set(b.get("files", {})))
           if a.get("files", {}).get(p) != b.get("files", {}).get(p)]
    out += [f"libero_plus.{k}" for k in LIBERO_CONTRACT_KEYS if a.get("libero_plus", {}).get(k) != b.get("libero_plus", {}).get(k)]
    return out
