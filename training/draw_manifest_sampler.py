#!/usr/bin/env python3
"""draw manifest を sampler へ差し込む（Codex 2026-09-23 裁定 §2）。

`EpisodeAwareSampler` の順序は `torch.randperm`（global RNG）で決まるので、Full と LoRA で
別の順序になる。ここでは **sampler クラスの `__init__` / `__iter__` / `__len__` を差し替え**、
凍結した global index の列をそのまま順に返させる。

  ・`__init__` の後で、manifest の全 index が sampler の候補（`self.indices`）に含まれるかを検査する。
    含まれなければ **RuntimeError**（別の dataset・別の episodes 指定で走っている）。
  ・`__iter__` は **1 回しか回らない**。2 回目の呼び出し（= `cycle()` が 1 周を使い切って再開した）は
    **RuntimeError**。step 数・GA・batch_size が 48,000 draw と噛み合っていないことを意味し、
    黙って同じ列を繰り返すと draw 数が変わるため止める。
  ・消費した index の数と running hash を記録し、両腕で**同じ列を消費したか**を後から照合できるようにする。
  ・`on_event`（wrapper は wrap JSON の `dump`）を **sampler 構築時・最初の 1 件を出す直前・最後の 1 件を出した後**に呼ぶ。
    これが無いと、sampler が構築されたか（＝フックが実際に効いたか）が学習終了まで wrap JSON に出ない。
    ★ 2026-09-23 の Mac 試運転で実際に起きた: lerobot は `hasattr(cfg.policy, "drop_n_last_frames")` の時だけ
    `EpisodeAwareSampler` を使い、無ければ普通の shuffle に落ちる。その場合 `__init__` が一度も呼ばれず
    manifest は**黙って素通り**され、`n_manifest=48000` だけが記録に残った。起動前ゲートは `sampler_candidates` と
    `iterations>=1` を待つこと（どちらも学習の最初の batch より前に確定し、この callback で書き出される）。

**lerobot の class オブジェクトの属性を差し替える**ので、`lerobot_train.py` が
`from lerobot.datasets.sampler import EpisodeAwareSampler` で先に名前を取り込んでいても効く（同じ class）。
"""
from __future__ import annotations

import hashlib


def install(sampler_cls, global_indices, record: dict | None = None, on_event=None) -> dict:
    gi = [int(x) for x in global_indices]
    state = record if record is not None else {}
    state.update({"n_manifest": len(gi), "consumed": 0, "iterations": 0,
                  "running_sha256": None, "first": gi[:4], "last": gi[-4:]})

    def _emit():
        if on_event is not None:
            try:
                on_event()
            except Exception:  # noqa: BLE001  記録の失敗で学習を止めない
                pass
    orig_init = sampler_cls.__init__

    def _init(self, *a, **k):
        orig_init(self, *a, **k)
        cand = set(int(x) for x in self.indices)
        missing = [g for g in gi if g not in cand]
        if missing:
            raise RuntimeError(f"draw manifest の index {len(missing)} 件が sampler の候補に無い"
                               f"（先頭 {missing[:3]}）。dataset か episodes 指定が manifest と違う")
        state["sampler_candidates"] = len(cand)
        _emit()

    def _iter(self):
        state["iterations"] += 1
        if state["iterations"] > 1:
            raise RuntimeError("draw manifest を使い切って 2 周目に入ろうとした。"
                               "--steps・GA・batch_size が manifest の draw 数と合っていない")
        h = hashlib.sha256()
        _emit()                                  # 1 件目を出す直前（iterations=1・consumed=0）
        for g in gi:
            h.update(g.to_bytes(8, "little"))
            state["consumed"] += 1
            state["running_sha256"] = h.hexdigest()
            yield g
        _emit()                                  # 全件を出し切った（consumed=n_manifest）

    def _len(self):
        return len(gi)

    sampler_cls.__init__ = _init
    sampler_cls.__iter__ = _iter
    sampler_cls.__len__ = _len
    return state
