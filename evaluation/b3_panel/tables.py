#!/usr/bin/env python3
"""B3/T3 の評価記録（`run_panel.py` の jsonl）の検査・G-ZERO マスクの凍結・全件表と G-ZERO 非違反 subset 表。Codex 2026-09-26 裁定 (b)。

  check  <run.jsonl> ...                          記録の検査だけ（重複・欠け・例外・照合項目・完走・runtime の一致）
  freeze --gzero <run.jsonl> --out <mask.json>    ゼロ行動の走行から G-ZERO マスクを凍結する（候補の結果を見る前に。上書きしない）
  tables --mask <mask.json> --runs <run.jsonl>... [--ref <label>] --out <dir>
                                                  候補ごとに場面の指紋を G-ZERO と照合し、全件表と subset 表を並べる
  pair   --a <run.jsonl> --b <run.jsonl> [--loose] 2 走行の場面の指紋と終端の一致（再現性の確認）。--loose は task×init だけで突き合わせる
                                                  （① と ③、seed 違いの比較。キーの他の項目は違ってよい）

契約（裁定 (b)）:
  ・主表は全件。subset は G-ZERO 非違反の感度分析で、公式の分母でも「方策では回避不能」の認定でもない。
  ・キー = manifest・task/BDDL・元 init と init state の sha・env seed・reset 方式・episode 順・env 内の reset 通番（reset_contract.KEY_FIELDS）。
    推論系列の seed は別の列。同じ task 名だけで複数 init を上書きしない。
  ・G-ZERO は候補の結果を見る前に凍結し、全候補で共用。後から成功が出た行だけマスクを変えない。
  ・場面の指紋（乱数状態・家具・init・静定後の状態）が G-ZERO と 1 つでも違う行があれば、その走行は昇格判定に使わない（promotion_blocked）。
  ・欠け・例外は失敗にも成功にも置き換えない（件数として出す）。micro と族 macro を別に出す。subset が空の族は NA。
  ・全件と subset で比較の向きが逆なら、単独で昇格を決めず、事前に固定した追加確認へ進む（表に「逆転」と出す）。
  ・区間は出していない（出すなら対応と task／族のまとまりを保つ方法を明記すること）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import reset_contract as rc  # noqa: E402


class TableError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- 読み込みと検査

def load_run(path: Path) -> dict[str, Any]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows or rows[0].get("kind") != "run_meta":
        raise TableError(f"{path}: 先頭が run_meta でない")
    meta = rows[0]
    return {"path": str(path), "sha256": rc.sha256_bytes(Path(path).read_bytes()), "meta": meta,
            "episodes": [r for r in rows if r.get("kind") == "episode"],
            "task_errors": [r for r in rows if r.get("kind") == "task_error"],
            "task_ends": [r for r in rows if r.get("kind") == "task_end"],
            "run_end": next((r for r in rows if r.get("kind") == "run_end"), None)}


def completed_tasks(run: dict[str, Any], want: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """task_end が ok で、予定の episode 数がそろった task だけ（途中の task の行は使わない）。"""
    ends = {e["task_id"]: e for e in (json.loads(l) for l in Path(run["path"]).read_text().splitlines() if l.strip())
            if e.get("kind") == "task_end"}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in run["episodes"]:
        by_task[e["task_id"]].append(e)
    return {tid: eps for tid, eps in by_task.items() if ends.get(tid, {}).get("status") == "ok" and len(eps) == want.get(tid)}


def remaining_tasks(spec: str) -> list[str]:
    """label=file1,file2 のどの試行でも完走していない task（manifest の順）。再開はこれを新しい試行で頭から走らせる。"""
    paths = (spec.split("=", 1)[1] if "=" in spec else spec).split(",")
    runs = [load_run(Path(p)) for p in paths]
    manifest = load_manifest(runs[0]["meta"])
    want = {t["task_id"]: len(t["episodes"]) for t in manifest["tasks"]}
    done: set[str] = set()
    for r in runs:
        if r["meta"]["manifest"]["sha256"] != runs[0]["meta"]["manifest"]["sha256"]:
            raise TableError(f"manifest が違う試行が混ざっている: {r['path']}")
        done |= set(completed_tasks(r, want))
    return [t["task_id"] for t in manifest["tasks"] if t["task_id"] not in done]


def load_candidate(spec: str) -> dict[str, Any]:
    """`path` か `label=path1,path2,...`。複数の試行は task 単位でまとめる（裁定 (a)5: 途中再開は task 単位の新しい試行で走らせ直す）。

    まとめ方: 各試行で task_end が ok かつ予定の episode がそろった task だけを使う（途中の task の行は使わない）。
    同じ task が 2 つ以上の試行で完走していたら止まる（どちらを使うかを後から選ばない）。条件（manifest・方式・seed・
    runtime・方策の記述子）が違う試行は混ぜない。
    """
    label, paths = (spec.split("=", 1) if "=" in spec else (None, spec))
    runs = [load_run(Path(p)) for p in paths.split(",")]
    if len(runs) == 1:
        if label:
            runs[0]["meta"] = {**runs[0]["meta"], "label": label}
        return runs[0]
    base = runs[0]["meta"]
    for r in runs[1:]:
        m = r["meta"]
        for f in ("mode", "env_seed", "runtime_fingerprint", "policy_descriptor"):
            if m.get(f) != base.get(f):
                raise TableError(f"まとめる試行の条件が違う（{f}）: {r['path']}")
        if m["manifest"]["sha256"] != base["manifest"]["sha256"]:
            raise TableError(f"まとめる試行の manifest が違う: {r['path']}")
    manifest = load_manifest(base)
    want = {t["task_id"]: len(t["episodes"]) for t in manifest["tasks"]}
    chosen: dict[str, tuple[int, list[dict[str, Any]]]] = {}
    for i, r in enumerate(runs):
        for tid, eps in completed_tasks(r, want).items():
            if tid in chosen:
                raise TableError(f"task {tid} が 2 つの試行で完走している（どちらを使うかを後から選ばない）")
            chosen[tid] = (i, eps)
    merged = [e for tid in (t["task_id"] for t in manifest["tasks"]) if tid in chosen for e in chosen[tid][1]]
    complete = set(chosen) == set(want)
    return {"path": ",".join(r["path"] for r in runs), "sha256": ",".join(r["sha256"] for r in runs),
            "meta": {**base, "label": label or base.get("label"), "task_filter": None,
                     "trial_id": "+".join(r["meta"]["trial_id"] for r in runs), "merged_from": [r["path"] for r in runs],
                     "merged_task_source": {tid: runs[i]["path"] for tid, (i, _) in chosen.items()}},
            "episodes": merged, "task_errors": [],
            "task_ends": [{"kind": "task_end", "task_id": tid, "status": "ok", "merged_from": runs[i]["path"]} for tid, (i, _) in chosen.items()],
            "run_end": {"kind": "run_end", "status": "complete" if complete else "incomplete_after_merge"}}


def load_manifest(meta: dict[str, Any]) -> dict[str, Any]:
    p = REPO / meta["manifest"]["path"]
    if rc.sha256_bytes(p.read_bytes()) != meta["manifest"]["sha256"]:
        raise TableError(f"manifest {p} の sha が走行時の記録と違う（凍結物が書き換わった）")
    return json.loads(p.read_text())


def planned_keys(manifest_sha: str, manifest: dict[str, Any], env_seed: int, mode: str) -> dict[tuple, dict[str, Any]]:
    """manifest から「実行されるべきキー」を作る。③ は task 内の通番 = episode_id＋1、① は毎回 1。"""
    out = {}
    for t in manifest["tasks"]:
        for e in t["episodes"]:
            ordinal = int(e["episode_id"]) + 1 if mode == "kit" else 1
            rec = {"manifest_sha256": manifest_sha, "task_stem": t["task_stem"], "bddl_sha256": t["bddl_sha256"],
                   "init_index": int(e["init_index"]), "init_state_sha256": e["init_state_sha256"], "env_seed": env_seed,
                   "reset_mode": mode, "episode_id": int(e["episode_id"]), "reset_ordinal_in_env": ordinal}
            out[rc.episode_key(rec)] = {"task": t, "episode": e}
    return out


def check_run(run: dict[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    """1 走行の検査。問題は problems に積む（空なら合格）。"""
    meta = run["meta"]
    problems: list[str] = []
    manifest = load_manifest(meta)
    planned = planned_keys(meta["manifest"]["sha256"], manifest, int(meta["env_seed"]), meta["mode"])
    if meta.get("task_filter"):
        want = set(meta["task_filter"])
        planned = {k: v for k, v in planned.items() if v["task"]["task_id"] in want}
    seen: dict[tuple, dict[str, Any]] = {}
    dup = []
    for e in run["episodes"]:
        k = rc.episode_key(e)
        if k in seen:
            dup.append(k)
        seen[k] = e
    extra = sorted(set(seen) - set(planned), key=str)
    missing = sorted(set(planned) - set(seen), key=str)
    bad_cross = [rc.episode_key(e) for e in run["episodes"] if not e.get("crosscheck_ok")]
    settle = {int(rc.dig(e, "settle.steps") or -1) for e in run["episodes"]}
    if dup:
        problems.append(f"同じキーの episode が重複: {len(dup)}")
    if extra:
        problems.append(f"manifest に無いキー: {len(extra)}")
    if missing:
        problems.append(f"欠け（予定にあるが記録なし）: {len(missing)}")
    if run["task_errors"]:
        problems.append(f"task の例外: {len(run['task_errors'])}")
    if bad_cross:
        problems.append(f"照合項目の不一致（crosscheck）: {len(bad_cross)}")
    if settle - {rc_settle()}:
        problems.append(f"静定 step 数が 10 でない episode がある: {sorted(settle)}")
    if require_complete and (run["run_end"] is None or run["run_end"].get("status") != "complete"):
        problems.append(f"走行が完走していない（run_end={run['run_end'] and run['run_end'].get('status')}）")
    return {"path": run["path"], "sha256": run["sha256"], "label": meta.get("label"), "trial_id": meta.get("trial_id"),
            "policy": meta.get("policy"), "mode": meta.get("mode"), "env_seed": meta.get("env_seed"),
            "runtime_fingerprint": meta.get("runtime_fingerprint"), "manifest_sha256": meta["manifest"]["sha256"],
            "planned": len(planned), "executed": len(seen), "missing": len(missing), "duplicates": len(dup), "extra": len(extra),
            "task_errors": len(run["task_errors"]), "crosscheck_failed": len(bad_cross), "problems": problems, "ok": not problems,
            "_planned": planned, "_seen": seen}


def rc_settle() -> int:
    return 10


def strip_private(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def same_conditions(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    out = []
    for f in ("manifest_sha256", "mode", "env_seed", "runtime_fingerprint"):
        if a[f] != b[f]:
            out.append(f)
    return out


# ----------------------------------------------------------------------------- G-ZERO の凍結

def violation_timing(e: dict[str, Any]) -> str:
    s = rc.dig(e, "nontarget.first_violation_step")
    if not e.get("collided"):
        return "none"
    if s is None:
        return "unknown"
    return "initial" if int(s) <= 1 else "later"


def freeze(gzero_path: Path, out: Path) -> dict[str, Any]:
    run = load_run(gzero_path)
    if run["meta"].get("policy") != "zero":
        raise TableError("G-ZERO はゼロ行動（--policy zero）の走行だけから作る")
    chk = check_run(run)
    if not chk["ok"]:
        raise TableError("G-ZERO の走行が検査に通らない: " + "; ".join(chk["problems"]))
    thr = float(run["meta"]["runtime"]["collision_threshold_m"])
    rows = []
    for k, e in sorted(chk["_seen"].items(), key=lambda kv: (kv[1]["task_order"], kv[1]["episode_id"])):
        rows.append({"key": dict(zip(rc.KEY_FIELDS, k)), "task_id": e["task_id"], "family": e.get("family"), "group": e.get("group"),
                     "collided": bool(e["collided"]), "done_raw": bool(e["done_raw"]), "termination": e["termination"],
                     "violation_timing": violation_timing(e), "first_violation_step": rc.dig(e, "nontarget.first_violation_step"),
                     "moved_mm": {n: round(v * 1000, 3) for n, v in (rc.dig(e, "nontarget.max_disp_m") or {}).items() if v > thr},
                     "fingerprint": {f: rc.dig(e, f) for f in rc.CORRESPONDENCE_FIELDS}})
    masked = [r for r in rows if r["collided"]]
    obj = {"kind": "gzero_mask", "schema": 1, "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "rule": "マスク = 同じ manifest・env seed・reset 方式・init・episode 順で、ゼロ行動でも衝突判定が立ったキー。"
                   "候補の結果を見る前に凍結し、全候補で共用する。subset は感度分析であり、公式の分母でも回避不能の認定でもない。",
           "gzero_run": {"path": str(Path(gzero_path).resolve().relative_to(REPO)), "sha256": run["sha256"],
                         "trial_id": run["meta"]["trial_id"], "label": run["meta"]["label"]},
           "manifest_sha256": chk["manifest_sha256"], "mode": chk["mode"], "env_seed": chk["env_seed"],
           "runtime_fingerprint": chk["runtime_fingerprint"], "runtime": run["meta"]["runtime"],
           "n": len(rows), "n_masked": len(masked),
           "masked_timing": {t: sum(1 for r in masked if r["violation_timing"] == t) for t in ("initial", "later", "unknown")},
           "rows": rows}
    body = {k: v for k, v in obj.items() if k != "created_at"}
    if out.exists():
        old = json.loads(out.read_text())
        if {k: v for k, v in old.items() if k != "created_at"} != body:
            raise TableError(f"凍結済みのマスク {out} と中身が違う。上書きしない（新しい版名で作ること）")
        print(f"既存と同一: {out}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
        print(f"凍結: {out}")
    print(f"sha256 {rc.sha256_bytes(out.read_bytes())}  n={len(rows)} masked={len(masked)} timing={obj['masked_timing']}")
    return obj


# ----------------------------------------------------------------------------- 表

def summarize(eps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(eps), "success": sum(bool(e["success"]) for e in eps), "collided": sum(bool(e["collided"]) for e in eps),
            "termination": rc.termination_counts(e["termination"] for e in eps)}


def macro(by_family: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rates = {f: (sum(bool(e["success"]) for e in eps) / len(eps) if eps else None) for f, eps in sorted(by_family.items())}
    have = [r for r in rates.values() if r is not None]
    return {"family_rates": rates, "families_with_rows": len(have), "families_empty": sorted(f for f, r in rates.items() if r is None),
            "macro": (sum(have) / len(have) if have else None)}


def contract_sidecar(mask_path: Path) -> dict[str, Any] | None:
    """凍結マスクの来歴の別紙（`mask_contract_sidecar.py` が作る v2）。無ければ None。"""
    p = Path(mask_path).with_name(Path(mask_path).stem + ".contract_v2.json")
    return json.loads(p.read_text()) if p.exists() else None


def promotion_problems(run: dict[str, Any], chk: dict[str, Any], mrows: dict[tuple, Any],
                       sidecar: dict[str, Any] | None) -> list[str]:
    """昇格判定の関門（Codex 2026-09-26 P1）。表示用の検査（task を絞った予定との比較）とは別に、**凍結した全キー**で見る。

    凍結マスクの全キーが 1 回ずつそろう・余分なキーなし・全 task が正常終端・走行の完了記録（または検証済みの task 単位の
    まとめ）・例外なし・照合項目が全て ok・task を絞った単独の走行でない・判定に効くコードが G-ZERO と同じ（別紙があれば）。
    """
    p: list[str] = []
    planned, seen = set(mrows), set(chk["_seen"])
    if planned - seen:
        p.append(f"凍結した {len(planned)} 行のうち欠け {len(planned - seen)}")
    if seen - planned:
        p.append(f"凍結した行に無いキー {len(seen - planned)}")
    if chk["duplicates"]:
        p.append(f"重複 {chk['duplicates']}")
    if run["task_errors"]:
        p.append(f"task の例外 {len(run['task_errors'])}")
    if chk["crosscheck_failed"]:
        p.append(f"照合項目の不一致 {chk['crosscheck_failed']}")
    tasks = {r["task_id"] for r in mrows.values()}
    ended = {e["task_id"] for e in run.get("task_ends", []) if e.get("status") == "ok"}
    if tasks - ended:
        p.append(f"正常に終わっていない task {len(tasks - ended)}")
    if (run.get("run_end") or {}).get("status") != "complete":
        p.append("走行の完了記録が無い（または完了でない）")
    if run["meta"].get("task_filter") and not run["meta"].get("merged_from"):
        p.append("task を絞った単独の走行（全 panel ではない）")
    if sidecar is not None:
        cf = run["meta"].get("contract_fingerprint")
        if cf is None:
            p.append("判定に効くコードの来歴（contract）が無い記録")
        elif cf != sidecar["contract_fingerprint"]:
            diff = rc.contract_diff(run["meta"].get("runtime_contract") or {}, sidecar["contract"])
            p.append("判定に効くコードが G-ZERO と違う: " + ", ".join(diff))
    return p


def build_tables(mask_path: Path, run_paths: list[Any], ref_label: str | None) -> dict[str, Any]:
    mask = json.loads(mask_path.read_text())
    mrows = {tuple(r["key"][f] for f in rc.KEY_FIELDS): r for r in mask["rows"]}
    masked = {k for k, r in mrows.items() if r["collided"]}
    sidecar = contract_sidecar(mask_path)
    res: dict[str, Any] = {"mask": {"path": str(mask_path), "sha256": rc.sha256_bytes(mask_path.read_bytes()), "n": mask["n"],
                                    "n_masked": mask["n_masked"], "masked_timing": mask["masked_timing"],
                                    "contract_sidecar": (sidecar or {}).get("contract_fingerprint")}, "runs": {}}
    per_run_eps: dict[str, dict[tuple, dict[str, Any]]] = {}
    for p in run_paths:
        run = load_candidate(str(p))
        chk = check_run(run, require_complete=False)
        label = chk["label"]
        if label in res["runs"]:
            raise TableError(f"同じ label の走行が 2 つある: {label}")
        cond = [f for f in ("manifest_sha256", "mode", "env_seed", "runtime_fingerprint") if chk[f] != mask[f]]
        mism = {}
        for k, e in chk["_seen"].items():
            if k not in mrows:
                mism[str(k)] = ["キーが G-ZERO に無い"]
                continue
            bad = [f for f in rc.CORRESPONDENCE_FIELDS
                   if rc.dig(e, f) is None or rc.dig(e, f) != rc.fingerprint_get(mrows[k]["fingerprint"], f)]
            if bad:
                mism[str(k)] = bad
        promo = promotion_problems(run, chk, mrows, sidecar)
        blocked = bool(cond or mism or chk["problems"] or promo)
        eps = list(chk["_seen"].items())
        full = [e for _, e in eps]
        sub = [e for k, e in eps if k not in masked]
        by_fam_full, by_fam_sub = defaultdict(list), defaultdict(list)
        for k, e in eps:
            by_fam_full[e.get("family") or e["task_id"]].append(e)
            by_fam_sub.setdefault(e.get("family") or e["task_id"], [])
            if k not in masked:
                by_fam_sub[e.get("family") or e["task_id"]].append(e)
        groups = sorted({e.get("group") for e in full})
        res["runs"][label] = {
            **strip_private(chk), "policy_descriptor": run["meta"].get("policy_descriptor"),
            "conditions_differ_from_mask": cond, "correspondence_mismatch": mism, "promotion_problems": promo,
            "promotion_blocked": blocked,
            "excluded_by_mask": sum(1 for k, _ in eps if k in masked),
            "masked_success": sum(bool(e["success"]) for k, e in eps if k in masked),
            "full": summarize(full), "subset": summarize(sub),
            "by_group": {g: {"full": summarize([e for e in full if e.get("group") == g]),
                             "subset": summarize([e for e in sub if e.get("group") == g])} for g in groups},
            "macro": {"full": macro(by_fam_full), "subset": macro(by_fam_sub)},
        }
        per_run_eps[label] = dict(eps)
    if ref_label:
        if ref_label not in per_run_eps:
            raise TableError(f"基準 {ref_label} が --runs に無い")
        R = per_run_eps[ref_label]
        comps = {}
        for label, E in per_run_eps.items():
            if label == ref_label:
                continue

            def diff(keys: list[tuple]) -> dict[str, Any]:
                """成功は safe_success（Codex 2026-09-26 §3）。衝突の増減も同じ行で数える。"""
                keys = [k for k in keys if k in R and k in E]
                ok = (lambda e: bool(e.get("safe_success", e["success"])))   # noqa: E731
                lost = sum(ok(R[k]) and not ok(E[k]) for k in keys)
                gained = sum(ok(E[k]) and not ok(R[k]) for k in keys)
                d = gained - lost
                return {"common_n": len(keys), "ref_success": sum(ok(R[k]) for k in keys), "cand_success": sum(ok(E[k]) for k in keys),
                        "lost": lost, "gained": gained, "diff": d, "rate_diff_pt": (100.0 * d / len(keys) if keys else None),
                        "collided_increase": sum(bool(E[k]["collided"]) for k in keys) - sum(bool(R[k]["collided"]) for k in keys)}

            common = sorted(set(R) & set(E), key=str)
            f, s = diff(common), diff([k for k in common if k not in masked])
            fam_of = {k: (E[k].get("family") or E[k]["task_id"]) for k in common}
            grp_of = {k: E[k].get("group") for k in common}
            by_family = {fam: diff([k for k in common if fam_of[k] == fam]) for fam in sorted(set(fam_of.values()))}
            by_group = {g: {"full": diff([k for k in common if grp_of[k] == g]),
                            "subset": diff([k for k in common if grp_of[k] == g and k not in masked])} for g in sorted({*grp_of.values()} - {None})}
            comps[label] = {"full": f, "subset": s, "by_group": by_group, "by_family": by_family,
                            "families_with_new_success": sorted(fam for fam, x in by_family.items() if x["gained"] > 0),
                            "reversal": (f["diff"] > 0 > s["diff"]) or (f["diff"] < 0 < s["diff"]),
                            "blocked": res["runs"][label]["promotion_blocked"] or res["runs"][ref_label]["promotion_blocked"]}
        res["vs_ref"] = {"ref": ref_label, "comparisons": comps}
    return res


def tables_md(res: dict[str, Any]) -> str:
    L = ["# B3/T3 評価の全件表と G-ZERO 非違反 subset 表（自動生成）", "",
         "主表は全件。subset は環境診断の感度分析で、公式の分母でも「方策では回避不能」の認定でもない。区間は出していない。",
         f"マスク: `{res['mask']['path']}`（sha {res['mask']['sha256'][:16]}…）除外 {res['mask']['n_masked']}/{res['mask']['n']}、"
         f"初回 step の違反 {res['mask']['masked_timing']['initial']}・後続 {res['mask']['masked_timing']['later']}・時刻不明 {res['mask']['masked_timing']['unknown']}"
         "（後続 step の違反は方策が先に成功・回避できる可能性を残す）", "",
         "| 走行 | 予定 / 実行 / 欠け / 例外 | 除外 | 全件 成功/n（衝突・時間切れ・done 後の衝突） | subset 成功/n（同） | 族 macro 全件 / subset（残る族） | 昇格判定 |",
         "|---|---|---|---|---|---|---|"]
    for label, r in res["runs"].items():
        def cell(t: dict[str, Any]) -> str:
            x = t["termination"]
            return f"{t['success']}/{t['n']}（{t['collided']}・{x[rc.TIMEOUT]}・{x[rc.DONE_COLLIDED]}）"
        mf, ms = r["macro"]["full"], r["macro"]["subset"]
        fmt = lambda v: "NA" if v is None else f"{100 * v:.1f}%"   # noqa: E731
        why = "使える" if not r["promotion_blocked"] else "★ 使えない（" + "・".join(
            (["条件が G-ZERO と違う: " + ",".join(r["conditions_differ_from_mask"])] if r["conditions_differ_from_mask"] else [])
            + ([f"場面の指紋の不一致 {len(r['correspondence_mismatch'])} 行"] if r["correspondence_mismatch"] else [])
            + list(dict.fromkeys(r["problems"] + r.get("promotion_problems", [])))) + "）"
        L.append(f"| {label} | {r['planned']} / {r['executed']} / {r['missing']} / {r['task_errors']} | {r['excluded_by_mask']} | "
                 f"{cell(r['full'])} | {cell(r['subset'])} | {fmt(mf['macro'])} / {fmt(ms['macro'])}（{ms['families_with_rows']}） | {why} |")
    if "vs_ref" in res:
        def dc(x: dict[str, Any]) -> str:
            rate = "NA" if x["rate_diff_pt"] is None else f"{x['rate_diff_pt']:+.2f}pt"
            return f"{x['diff']:+d}（−{x['lost']} ＋{x['gained']}）/ {x['common_n']}・{rate}・衝突 {x['collided_increase']:+d}"
        L += ["", f"基準 {res['vs_ref']['ref']} との対応比較（候補 − 基準、成功は safe_success、共通の行だけ。欠けは置き換えない）:", "",
              "| 候補 | 全件 差（−失う ＋得る）/ 共通 n・率差・衝突増 | subset 差（同） | learned 全件 | holdout 全件 | 新規成功の族 | 注意 |",
              "|---|---|---|---|---|---|---|"]
        for label, c in res["vs_ref"]["comparisons"].items():
            note = "・".join((["★ 全件と subset で向きが逆（保留）"] if c["reversal"] else [])
                             + (["★ 昇格判定に使えない走行を含む"] if c["blocked"] else [])) or "—"
            g = c["by_group"]
            gl = dc(g["learned"]["full"]) if "learned" in g else "—"
            gh = dc(g["holdout"]["full"]) if "holdout" in g else "—"
            L.append(f"| {label} | {dc(c['full'])} | {dc(c['subset'])} | {gl} | {gh} | {len(c['families_with_new_success'])} | {note} |")
        for label, c in res["vs_ref"]["comparisons"].items():
            L += ["", f"族ごと（{label} − {res['vs_ref']['ref']}）:", "",
                  "| 族 | n | 基準 成功 | 候補 成功 | 失う | 得る | 衝突増 |", "|---|---|---|---|---|---|---|"]
            for fam, x in c["by_family"].items():
                L.append(f"| {fam} | {x['common_n']} | {x['ref_success']} | {x['cand_success']} | {x['lost']} | {x['gained']} | {x['collided_increase']:+d} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- 再現性の確認

OUTCOME_FIELDS = ("done_raw", "collided", "success", "steps", "termination", "nontarget.max_disp_m", "nontarget.first_cross_step")


def pair(a_path: Path, b_path: Path, loose: bool) -> dict[str, Any]:
    A, B = load_run(a_path), load_run(b_path)

    def key(e: dict[str, Any]) -> tuple:
        return (e["task_stem"], e["init_index"], e["init_state_sha256"]) if loose else rc.episode_key(e)

    ea, eb = {key(e): e for e in A["episodes"]}, {key(e): e for e in B["episodes"]}
    rows = []
    for k in sorted(set(ea) & set(eb), key=str):
        x, y = ea[k], eb[k]
        rows.append({"task_id": x["task_id"], "init_index": x["init_index"],
                     "scene_diff": [f for f in rc.CORRESPONDENCE_FIELDS if rc.dig(x, f) != rc.dig(y, f)],
                     "outcome_diff": [f for f in OUTCOME_FIELDS if rc.dig(x, f) != rc.dig(y, f)],
                     "fixtures_a": rc.dig(x, "reset.fixtures_digest"), "fixtures_b": rc.dig(y, "reset.fixtures_digest"),
                     "fixture_max_abs_diff_mm": fixture_max_diff_mm(x, y),
                     "termination": [x["termination"], y["termination"]], "steps": [x["steps"], y["steps"]]})
    return {"a": {"path": str(a_path), "label": A["meta"]["label"], "mode": A["meta"]["mode"], "env_seed": A["meta"]["env_seed"],
                  "policy": A["meta"]["policy"], "runtime_fingerprint": A["meta"]["runtime_fingerprint"]},
            "b": {"path": str(b_path), "label": B["meta"]["label"], "mode": B["meta"]["mode"], "env_seed": B["meta"]["env_seed"],
                  "policy": B["meta"]["policy"], "runtime_fingerprint": B["meta"]["runtime_fingerprint"]},
            "loose": loose, "only_a": len(set(ea) - set(eb)), "only_b": len(set(eb) - set(ea)), "n": len(rows),
            "scene_identical": sum(1 for r in rows if not r["scene_diff"]),
            "outcome_identical": sum(1 for r in rows if not r["outcome_diff"]), "rows": rows}


def fixture_max_diff_mm(x: dict[str, Any], y: dict[str, Any]) -> float | None:
    fa, fb = rc.dig(x, "reset.fixtures") or {}, rc.dig(y, "reset.fixtures") or {}
    if set(fa) != set(fb):
        return None
    m = 0.0
    for n in fa:
        for i in range(3):
            m = max(m, abs(fa[n]["pos"][i] - fb[n]["pos"][i]))
    return round(m * 1000, 3)


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check")
    c.add_argument("runs", type=Path, nargs="+")
    f = sp.add_parser("freeze")
    f.add_argument("--gzero", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    t = sp.add_parser("tables")
    t.add_argument("--mask", type=Path, required=True)
    t.add_argument("--runs", nargs="+", required=True, help="path か label=path1,path2（途中再開した試行を task 単位でまとめる）")
    t.add_argument("--ref", default=None)
    t.add_argument("--out", type=Path, required=True)
    cm = sp.add_parser("check-merged", help="label=file1,file2 を task 単位でまとめ、凍結マスクの全キーで昇格の関門を通るか（通れば 0）")
    cm.add_argument("--spec", required=True)
    cm.add_argument("--mask", type=Path, required=True)
    rm_ = sp.add_parser("remaining", help="どの試行でも完走していない task_id を空白区切りで出す")
    rm_.add_argument("--spec", required=True)
    p = sp.add_parser("pair")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--loose", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            res = [strip_private(check_run(load_run(r))) for r in a.runs]
            print(json.dumps(res, ensure_ascii=False, indent=1))
            return 0 if all(r["ok"] for r in res) else 1
        if a.cmd == "freeze":
            freeze(a.gzero, a.out)
            return 0
        if a.cmd == "check-merged":
            r = next(iter(build_tables(a.mask, [a.spec], None)["runs"].values()))
            print(json.dumps({k: r[k] for k in ("label", "planned", "executed", "missing", "promotion_blocked", "promotion_problems",
                                                "problems", "conditions_differ_from_mask")}, ensure_ascii=False, indent=1))
            return 0 if not r["promotion_blocked"] else 1
        if a.cmd == "remaining":
            print(" ".join(remaining_tasks(a.spec)))
            return 0
        if a.cmd == "tables":
            res = build_tables(a.mask, a.runs, a.ref)
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
            (a.out / "TABLES.md").write_text(tables_md(res))
            print(tables_md(res))
            return 0 if not any(r["promotion_blocked"] for r in res["runs"].values()) else 1
        res = pair(a.a, a.b, a.loose)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        for r in res["rows"]:
            print(f"  {r['task_id']} init{r['init_index']}: scene_diff={r['scene_diff']} outcome_diff={r['outcome_diff']} "
                  f"家具の最大差 {r['fixture_max_abs_diff_mm']} mm 終端 {r['termination']} step {r['steps']}")
        return 0
    except TableError as exc:
        print(f"[tables] ★ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
