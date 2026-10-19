#!/usr/bin/env python3
"""学習が**実際に**行った更新と食べたサンプルを記録し、最初の数 update で期待と照合する（Codex 2026-09-23 レビュー §3 A・D）。

draw manifest の `consumed` は sampler が DataLoader に渡した index の数で、**先読み（prefetch）を含む**。
それだけでは「optimizer が 48k を学習した」とは言えない。ここでは次を学習プロセスの中で数える:

  ・**実 update 数**: torch optimizer の step 後フック。accelerate の AcceleratedOptimizer は同期 micro-step でだけ
    中の optimizer.step を呼ぶので、GA=2 なら 2 micro-batch に 1 回だけ増える。
  ・**学習した sample**: `update_policy` に渡る batch の `index`（dataset の行番号）。micro-batch 数・sample 数・running sha。
    manifest の期待列（global index の並び）と**先頭から順に**一致するかも照合する。
  ・**最初の K update の照合**（既定 K=3）: 各 update が GA 個の micro-batch・実効 batch 個の sample からなること、
    sample ID が manifest の先頭と一致すること、PAD mask が全 micro-batch で適用されたこと（PAD を使う走りのみ）。
    **1 つでも外れたら RuntimeError で学習をその場で止める**（数分で分かる。90 分後の post_check まで待たない）。
    結果は wrap JSON の `early_check` に書く（学習の最初の数 update で確定する値であり、学習中ずっと更新されない値を待つゲートではない）。
  ・**勾配クリップを同期時だけにする設定の照合**（check_clip_sync=True のとき。Codex 2026-09-24 裁定 1）:
    wrapper の clip フックが入り、sync_unknown=0、同期時の呼び出し回数＝実 update 数、呼び出し総数＝micro-batch 数、
    GA>1 なら同期前の呼び出しが実際にあり（micro−update 回）その全件がスキップ、GA=1 なら同期前の呼び出しが 0。
    1 つでも外れたら early_check 不合格で止まる（フックが効いていない・同期状態が取れない走りを通さない）。
  ・**checkpoint の保存完了**: lerobot の `save_checkpoint` が返った直後（全ファイルを書き終えた後）に `READY.json` を
    一時ファイル → rename で原子的に書く。**step の表示・ディレクトリ・config の存在は保存完了の証拠にしない**（export 側は READY を待つ）。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _to_ints(x) -> list[int]:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().to("cpu").reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        pass
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


class TrainAudit:
    def __init__(self, info: dict, dump, expected=None, expect_ga=None, expect_eff=None,
                 early_k: int = 3, check_pad: bool = False, pad_info: dict | None = None, check_clip_sync: bool = False):
        self.info, self.dump = info, dump
        self.expected = list(expected) if expected is not None else None
        self.ga, self.eff, self.k = expect_ga, expect_eff, int(early_k)
        self.check_pad, self.pad_info = check_pad, pad_info
        self.check_clip_sync = bool(check_clip_sync)
        self.updates = self.micro = self.samples = 0
        self.m_since = self.s_since = 0
        self.per_update: list[tuple[int, int]] = []
        self.first_updates: list[list[int]] = []
        self._cur: list[int] = []
        self.h = hashlib.sha256()
        self.seq_ok = True
        self.first_mismatch = None
        self.missing_index = 0
        self.early_done = False
        self._publish()

    # ---- 記録
    def _publish(self):
        self.info["trained"] = {"updates": self.updates, "micro_batches": self.micro, "samples": self.samples,
                                "sha256": self.h.hexdigest(), "seq_matches_manifest": self.seq_ok if self.expected is not None else None,
                                "first_mismatch": self.first_mismatch, "missing_index_batches": self.missing_index}

    def on_optimizer_step(self, *_a, **_k):
        self.updates += 1

    def attach_optimizer(self, opt) -> bool:
        if hasattr(opt, "register_step_post_hook"):
            opt.register_step_post_hook(self.on_optimizer_step)
            self.info["trained_hook"] = "torch.optim.Optimizer.register_step_post_hook"
            return True
        self.info["trained_hook"] = "unavailable"
        return False

    def before_micro(self, batch) -> int:
        idx = batch.get("index") if isinstance(batch, dict) else None
        if idx is None:
            self.missing_index += 1
            ids: list[int] = []
        else:
            ids = _to_ints(idx)
        for g in ids:
            pos = self.samples
            if self.expected is not None and self.seq_ok:
                if pos >= len(self.expected) or self.expected[pos] != g:
                    self.seq_ok = False
                    self.first_mismatch = {"position": pos, "got": g,
                                           "expected": (self.expected[pos] if pos < len(self.expected) else None)}
            self.h.update(int(g).to_bytes(8, "little", signed=True))
            self.samples += 1
        self.micro += 1
        self.m_since += 1
        self.s_since += len(ids)
        if len(self.per_update) < self.k:
            self._cur.extend(ids)
        return self.updates

    def after_micro(self, updates_before: int):
        if self.updates > updates_before:
            if len(self.per_update) < self.k:
                self.per_update.append((self.m_since, self.s_since))
                self.first_updates.append(list(self._cur))
                self._cur = []
            self.m_since = self.s_since = 0
        self._publish()
        if not self.early_done and len(self.per_update) >= self.k:
            self.early_done = True
            self._early_check()

    def _early_check(self):
        bad = []
        for i, (m, s) in enumerate(self.per_update):
            if self.ga is not None and m != self.ga:
                bad.append(f"update {i}: micro-batch {m} != GA {self.ga}")
            if self.eff is not None and s != self.eff:
                bad.append(f"update {i}: sample {s} != 実効 batch {self.eff}")
        if self.missing_index:
            bad.append(f"batch に index が無い micro-batch が {self.missing_index} 個（sample ID を照合できない）")
        if self.expected is not None and not self.seq_ok:
            bad.append(f"学習した sample が manifest の先頭と違う: {self.first_mismatch}")
        if self.check_pad:
            applied = (self.pad_info or {}).get("applied")
            if applied != self.micro:
                bad.append(f"PAD mask の適用 {applied} != micro-batch {self.micro}")
        clip_rec = None
        if self.check_clip_sync:
            clip_rec, clip_bad = self._clip_problems()
            bad.extend(clip_bad)
        rec = {"ok": not bad, "k_updates": self.k, "per_update_micro_samples": self.per_update,
               "first_updates_indices": self.first_updates, "expect_ga": self.ga, "expect_eff": self.eff,
               "pad_checked": self.check_pad, "clip_sync_checked": self.check_clip_sync, "clip_at_check": clip_rec, "problems": bad}
        self.info["early_check"] = rec
        try:
            self.dump()
        finally:
            print("WRAP early_check", json.dumps({k: v for k, v in rec.items() if k != "first_updates_indices"}, ensure_ascii=False), flush=True)
        if bad:
            raise RuntimeError(f"TRAIN_AUDIT_EARLY_FAIL: {bad}")

    def _clip_problems(self):
        ci = self.info.get("grad_clip")
        bad = []
        if self.info.get("grad_clip_hook_error"):
            bad.append(f"clip のフックが入らなかった: {self.info.get('grad_clip_hook_error')}")
        if not isinstance(ci, dict) or ci.get("hook_installed") is not True:
            bad.append(f"clip のフックが入っていない: {ci}")
            return (dict(ci) if isinstance(ci, dict) else None), bad
        snap = dict(ci)
        if ci.get("only_on_sync") is not True:
            bad.append(f"clip の同期時のみ設定が効いていない: only_on_sync={ci.get('only_on_sync')}")
        if ci.get("sync_unknown") != 0:
            bad.append(f"同期状態が取れない clip 呼び出し sync_unknown={ci.get('sync_unknown')}")
        if ci.get("calls_when_sync") != self.updates:
            bad.append(f"同期時の clip 回数 {ci.get('calls_when_sync')} != 実 update {self.updates}")
        if ci.get("calls") != self.micro:
            bad.append(f"clip の呼び出し総数 {ci.get('calls')} != micro-batch {self.micro}（呼ばれない micro-step がある）")
        want_ns = self.micro - self.updates
        if self.ga is not None and self.ga > 1 and want_ns <= 0:
            bad.append(f"GA={self.ga} なのに同期前の micro-step が無い（micro {self.micro}・update {self.updates}）")
        if ci.get("calls_when_not_sync") != want_ns:
            bad.append(f"同期前の clip 回数 {ci.get('calls_when_not_sync')} != micro−update {want_ns}")
        if ci.get("skipped_not_sync") != ci.get("calls_when_not_sync"):
            bad.append(f"同期前の clip {ci.get('calls_when_not_sync')} 回のうちスキップ {ci.get('skipped_not_sync')} 回（全件スキップでない）")
        return snap, bad

    # ---- checkpoint の保存完了
    def mark_ready(self, checkpoint_dir, step):
        d = Path(checkpoint_dir)
        files = {}
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name not in ("READY.json", ".READY.json.tmp"):
                files[str(p.relative_to(d))] = p.stat().st_size
        self._publish()
        rec = {"step": int(step), "n_files": len(files), "bytes": sum(files.values()), "files": files, "trained": dict(self.info["trained"])}
        tmp = d / ".READY.json.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        os.replace(tmp, d / "READY.json")
        self.info.setdefault("checkpoints", []).append({"step": int(step), "dir": str(d), "n_files": len(files), "bytes": rec["bytes"],
                                                        "updates": self.updates, "samples": self.samples})
        self.dump()

    # ---- lerobot へ差し込む
    def install(self, lt_module):
        orig_upd = lt_module.update_policy

        def update_policy(*a, **k):
            batch = a[2] if len(a) > 2 else k.get("batch")
            before = self.before_micro(batch)
            r = orig_upd(*a, **k)
            self.after_micro(before)
            return r

        lt_module.update_policy = update_policy
        orig_save = lt_module.save_checkpoint

        def save_checkpoint(*a, **k):
            r = orig_save(*a, **k)
            d = k.get("checkpoint_dir", a[0] if a else None)
            s = k.get("step", a[1] if len(a) > 1 else None)
            self.mark_ready(d, s)
            return r

        lt_module.save_checkpoint = save_checkpoint
        self.info["train_audit"] = {"installed": True, "expect_ga": self.ga, "expect_eff": self.eff, "early_k": self.k,
                                    "check_pad": self.check_pad, "check_clip_sync": self.check_clip_sync, "expected_len": len(self.expected) if self.expected is not None else None}


def expected_from_manifest(doc: dict) -> list[int]:
    """manifest から「学習が食べるべき dataset の行番号（index 列）」の並びを取る。B3 は global＝sampler、B4 は draws の global_index。"""
    if "draws" in doc:
        return [int(d["global_index"]) for d in doc["draws"]]
    return [int(g) for g in doc["global_indices"]]


def expected_sha256(doc: dict) -> str:
    h = hashlib.sha256()
    for g in expected_from_manifest(doc):
        h.update(int(g).to_bytes(8, "little", signed=True))
    return h.hexdigest()
