#!/usr/bin/env python3
"""F0/F1 用の lerobot-train ラッパ（0.4.4）。学習本体は変えず、**実際に適用された設定を記録する**。

記録するもの（PI05_WRAP_LOG に JSON）:
  - accelerate の mixed_precision / native_amp / gradient_accumulation_steps（環境変数が本当に効いたか）
  - policy 構築直後の parameter dtype 別の個数・trainable 数、参照重み（PI05_REF_WEIGHTS）との一致（ロード同一性）
  - optimizer の種類・param dtype・foreach/fused
  - scheduler の解決済み設定（warmup/decay/peak/floor・主要 step の LR・build のソース sha）【F1 で追加】
  - 定常域の s/step（step 10 以降の壁時計）
  - 終了時の torch.cuda.max_memory_reserved / max_memory_allocated（peak reserved が正本）
  - **PI05_EXPECT_NORM があれば、実 normalizer / unnormalizer をその期待値と照合し、不一致なら学習を止める**【T2 で追加】
副作用は 3 つだけ:
  1. PI05_TOKENIZER_DIR があれば、`google/paligemma-3b-pt-224` のトークナイザ取得をローカル dir へ向ける
     （gated repo を Hub から引かないため）。提出物には一切関係しない（F0 の学習環境の道具）。
  2. PI05_EXPECT_NORM があって照合が不一致なら RuntimeError で止める（PI05_EXPECT_NORM_STRICT=0 で警告に落とせる）。
  3. PI05_FORCE_NORM があれば、lerobot へ渡る stats の state/action ブロックをその値へ差し替える（代替経路）。
  **どの環境変数も渡さなければ従来どおりの挙動**（F0/F1/T3-2 の走りは一切変わらない）。

PI05_EXPECT_NORM に渡せるのは:
  - normalizer の safetensors（`action.q01` のようなフラットキー。例: F1 の
    `policy_preprocessor_step_2_normalizer_processor.safetensors`）
  - 同じ値を入れ子 JSON にしたもの（`work/data/make_t2_stats_from_f1.py --emit-expect` の出力）
照合対象は `observation.state` と `action`。`count` は metadata なので報告のみで門にしない。

正規経路（推奨）は `work/data/make_t2_stats_from_f1.py --write` でデータセットの `meta/stats.json` を直し、
PI05_EXPECT_NORM で「本当に効いたか」を門にすること。PI05_FORCE_NORM は stats.json を書けない時の代替。
両方同時に使ってもよい（FORCE で入れて EXPECT で検算する）。
"""
import atexit, json, os, sys, time

LOG = os.environ.get("PI05_WRAP_LOG")
INFO = {"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in ("ACCELERATE_MIXED_PRECISION", "LEROBOT_GRAD_ACCUM", "ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "PI05_TOKENIZER_DIR", "HF_HUB_OFFLINE", "PI05_REF_WEIGHTS", "PI05_EXPECT_NORM", "PI05_EXPECT_NORM_STRICT", "PI05_FORCE_NORM", "PI05_DRAW_MANIFEST", "PI05_TRAIN_AUDIT", "PI05_EXPECT_GA", "PI05_EXPECT_EFF", "PI05_EARLY_UPDATES", "PI05_CLIP_ONLY_ON_SYNC")}}
def dump():
    # 2026-09-24: 一時ファイル → rename で原子的に書く（台本が学習中に wrap JSON を読むため。書きかけを読ませない）
    if LOG:
        try:
            tmp = f"{LOG}.tmp"
            with open(tmp, "w") as f:
                json.dump(INFO, f, indent=1, default=str)
            os.replace(tmp, LOG)
        except Exception as e:  # noqa: BLE001
            print("WRAP dump failed", e)

# --- tokenizer redirect（gated repo をローカルへ） ---
TOK = os.environ.get("PI05_TOKENIZER_DIR")
if TOK:
    import transformers
    def _redirect(cls_attr):
        orig = getattr(transformers, cls_attr).from_pretrained.__func__
        def fp(cls, name, *a, **k):
            if isinstance(name, str) and "paligemma-3b-pt-224" in name:
                INFO.setdefault("tokenizer_redirect", []).append(f"{cls_attr}:{name}->{TOK}")
                name = TOK
            return orig(cls, name, *a, **k)
        setattr(getattr(transformers, cls_attr), "from_pretrained", classmethod(fp))
    for attr in ("AutoTokenizer", "AutoProcessor"):
        try:
            _redirect(attr)
        except Exception as e:  # noqa: BLE001
            INFO[f"redirect_{attr}_error"] = str(e)

import torch  # noqa: E402

# --- accelerate hook ---
from accelerate import Accelerator  # noqa: E402
_acc_init = Accelerator.__init__
def _init(self, *a, **k):
    _acc_init(self, *a, **k)
    INFO["mixed_precision"] = str(self.mixed_precision)
    INFO["native_amp"] = bool(getattr(self, "native_amp", False))
    INFO["grad_accum_steps"] = int(getattr(self, "gradient_accumulation_steps", 1))
    print(f"WRAP accelerator mixed_precision={INFO['mixed_precision']} native_amp={INFO['native_amp']} grad_accum={INFO['grad_accum_steps']}")
    dump()
Accelerator.__init__ = _init

# --- policy hook（factory を差し替えてから lerobot_train を import する） ---
import lerobot.policies.factory as _pf  # noqa: E402
_make_policy = _pf.make_policy
def _stats(policy):
    from collections import Counter
    dt_all, dt_tr = Counter(), Counter(); n_tr = 0; n_all = 0
    for p in policy.parameters():
        dt_all[str(p.dtype)] += p.numel(); n_all += p.numel()
        if p.requires_grad:
            dt_tr[str(p.dtype)] += p.numel(); n_tr += p.numel()
    INFO["params_total"] = n_all; INFO["trainable_params"] = n_tr
    INFO["param_dtypes"] = dict(dt_all); INFO["trainable_dtypes"] = dict(dt_tr)
    ref = os.environ.get("PI05_REF_WEIGHTS")
    if ref and os.path.isfile(ref):
        from safetensors import safe_open
        sd = policy.state_dict(); n_cmp = n_eq = 0; maxd = 0.0; missing = 0
        with safe_open(ref, "pt") as f:
            keys = list(f.keys())
            sample = [k for i, k in enumerate(keys) if i % 25 == 0 or "action" in k or "state_proj" in k]
            for k in sample:
                if k not in sd:
                    missing += 1; continue
                a = f.get_tensor(k).to(torch.float32); b = sd[k].detach().to("cpu", torch.float32)
                if a.shape != b.shape:
                    missing += 1; continue
                d = float((a - b).abs().max()); maxd = max(maxd, d); n_cmp += 1; n_eq += int(d == 0.0)
        INFO["load_check"] = {"compared": n_cmp, "equal": n_eq, "max_abs_diff": maxd, "missing_or_shape": missing}
        INFO["load_identical"] = bool(n_cmp > 0 and n_eq == n_cmp and missing == 0)
        print(f"WRAP load_check compared={n_cmp} equal={n_eq} max_abs_diff={maxd} missing={missing} load_identical={INFO['load_identical']}")
    print(f"WRAP params total={n_all} trainable={n_tr} dtypes={dict(dt_all)} trainable_dtypes={dict(dt_tr)}")
    dump()
def make_policy(*a, **k):
    p = _make_policy(*a, **k)
    try:
        _stats(p)
    except Exception as e:  # noqa: BLE001
        INFO["stats_error"] = str(e); print("WRAP stats error", e)
    return p
_pf.make_policy = make_policy

# --- T3-2（Codex 2026-09-07 問 1）: 実 processor の正規化方式・統計・有限性を記録し、渡した dataset stats と照合する ---
_make_pp = _pf.make_pre_post_processors
def _find_normalizer(proc):
    steps = getattr(proc, "steps", None) or getattr(proc, "_steps", None) or []
    for st in steps:
        if "normaliz" in type(st).__name__.lower() and hasattr(st, "stats"):
            return st
    return None
def _norm_record(st, ds_stats):
    import numpy as np
    rec = {"class": type(st).__name__, "features": {}, "ok": True, "problems": []}
    nm = getattr(st, "norm_map", None) or {}
    rec["norm_map"] = {str(k): str(v) for k, v in dict(nm).items()}
    stats = getattr(st, "stats", None) or {}
    for key in ("observation.state", "action"):
        s = stats.get(key) if isinstance(stats, dict) else None
        d = (ds_stats or {}).get(key) if isinstance(ds_stats, dict) else None
        f = {"present": s is not None}
        if s is None:
            rec["ok"] = False; rec["problems"].append(f"{key}: stats 無し"); rec["features"][key] = f; continue
        for q in ("q01", "q99", "mean", "std"):
            v = s.get(q)
            if v is None:
                if q in ("q01", "q99"): rec["ok"] = False; rec["problems"].append(f"{key}: {q} 欠落")
                continue
            arr = np.asarray(v, dtype=float); f[q] = [round(float(x), 5) for x in arr.ravel()[:16]]
            if not np.all(np.isfinite(arr)): rec["ok"] = False; rec["problems"].append(f"{key}: {q} 非有限")
            if d is not None and d.get(q) is not None and not np.allclose(arr, np.asarray(d.get(q), dtype=float), atol=1e-6, rtol=1e-5):
                rec["ok"] = False; rec["problems"].append(f"{key}: {q} が dataset stats と不一致")
        if f.get("q01") and f.get("q99"):
            span = np.asarray(s["q99"], float) - np.asarray(s["q01"], float)
            if np.any(span <= 1e-6): rec["ok"] = False; rec["problems"].append(f"{key}: q99−q01≤0 の次元 {np.where(span<=1e-6)[0].tolist()}")
        rec["features"][key] = f
    return rec
# --- T2 本命（2026-09-09）: PI05_EXPECT_NORM があれば「実 normalizer が F1 と一致すること」を起動前に検査する ---
# 背景: lerobot 0.4.4 は pretrained を渡していても preprocessor_overrides で dataset の meta/stats.json に
#       **全置換**する（lerobot_train.py 258-281 / normalize_processor.py 196-201）。
#       単因子実験にするため、学習に使う stats を F1 と同一に固定したうえで、それが本当に効いたかをここで確かめる。
_EXPECT_PATH = os.environ.get("PI05_EXPECT_NORM")
_EXPECT_STRICT = str(os.environ.get("PI05_EXPECT_NORM_STRICT", "1")).lower() not in ("0", "false", "no", "off")
_EXPECT_KEYS = ("observation.state", "action")
_EXPECT_GATE_STATS = ("q01", "q99", "q10", "q50", "q90", "mean", "std", "min", "max")  # count は門にしない


def _load_expect(path):
    """safetensors（フラットキー `action.q01`）か JSON（入れ子）から期待統計を読む。"""
    import numpy as np
    out = {}
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, "np") as f:
            for flat in f.keys():
                key, stat = flat.rsplit(".", 1)
                out.setdefault(key, {})[stat] = np.asarray(f.get_tensor(flat), dtype=float)
    else:
        with open(path) as f:
            raw = json.load(f)
        for key, blk in raw.items():
            if key.startswith("_") or not isinstance(blk, dict):
                continue  # `_source` のような注記行は飛ばす
            out[key] = {s: np.asarray(v, dtype=float) for s, v in blk.items() if isinstance(v, (list, tuple))}
    return out


def _expect_compare(st, expect, tag):
    """実 processor step の stats を期待値と突き合わせる。戻り値 = 記録用 dict。"""
    import numpy as np
    rec = {"tag": tag, "match": True, "problems": [], "checked": 0, "count": {}}
    live = getattr(st, "stats", None) or {} if st is not None else {}
    if st is None:
        rec["match"] = False; rec["problems"].append(f"{tag}: normalizer step が無い"); return rec
    for key in _EXPECT_KEYS:
        if key not in expect:
            continue
        if key not in live:
            # unnormalizer は action しか持たない構成があるので、その欠落は問題にしない
            if tag == "unnormalizer" and key != "action":
                continue
            rec["match"] = False; rec["problems"].append(f"{tag}/{key}: 実 stats に無い"); continue
        for stat in _EXPECT_GATE_STATS:
            if stat not in expect[key]:
                continue
            if stat not in live[key]:
                rec["match"] = False; rec["problems"].append(f"{tag}/{key}.{stat}: 実 stats に無い"); continue
            got = np.asarray(live[key][stat], dtype=float).ravel()
            exp = np.asarray(expect[key][stat], dtype=float).ravel()
            if got.shape != exp.shape:
                rec["match"] = False
                rec["problems"].append(f"{tag}/{key}.{stat}: shape {list(got.shape)} != 期待 {list(exp.shape)}")
                continue
            rec["checked"] += 1
            if not np.allclose(got, exp, atol=1e-6, rtol=1e-5):
                rec["match"] = False
                d = float(np.abs(got - exp).max())
                rec["problems"].append(f"{tag}/{key}.{stat}: 最大差 {d:.3e}（実 {got[:4].tolist()} / 期待 {exp[:4].tolist()}）")
        if "count" in live[key]:
            rec["count"][key] = {"live": float(np.asarray(live[key]["count"], dtype=float).ravel()[0]),
                                 "expect": (float(np.asarray(expect[key]["count"], dtype=float).ravel()[0])
                                            if "count" in expect[key] else None)}
    if rec["checked"] == 0 and not rec["problems"]:
        rec["match"] = False; rec["problems"].append(f"{tag}: 照合できた統計が 0 件")
    return rec


_FORCE_PATH = os.environ.get("PI05_FORCE_NORM")


def _force_stats_into(kwargs, force):
    """【代替経路】meta/stats.json を書き換えられない時に、lerobot へ渡る stats を wrapper 側で差し替える。

    lerobot_train.py は `dataset_stats` と `preprocessor_overrides.normalizer_processor.stats` と
    `postprocessor_overrides.unnormalizer_processor.stats` の 3 か所へ `dataset.meta.stats` を渡す。
    その 3 か所の `observation.state` / `action` ブロックだけを差し替える（画像や index 系は触らない）。
    元の dict は共有オブジェクトなので **in-place で壊さず浅いコピーを作る**。
    """
    import numpy as np
    touched = []

    def patch(stats):
        if not isinstance(stats, dict):
            return stats, []
        out = dict(stats); done = []
        for key in _EXPECT_KEYS:
            if key not in force:
                continue
            base = dict(out.get(key) or {})
            for stat, val in force[key].items():
                base[stat] = np.asarray(val, dtype=np.float32)
            out[key] = base; done.append(key)
        return out, done

    if isinstance(kwargs.get("dataset_stats"), dict):
        kwargs["dataset_stats"], d = patch(kwargs["dataset_stats"]); touched += [f"dataset_stats:{x}" for x in d]
    ov = kwargs.get("preprocessor_overrides")
    if isinstance(ov, dict) and isinstance(ov.get("normalizer_processor"), dict):
        ov = dict(ov); npz = dict(ov["normalizer_processor"])
        npz["stats"], d = patch(npz.get("stats"))
        ov["normalizer_processor"] = npz; kwargs["preprocessor_overrides"] = ov
        touched += [f"normalizer_processor:{x}" for x in d]
    po = kwargs.get("postprocessor_overrides")
    if isinstance(po, dict) and isinstance(po.get("unnormalizer_processor"), dict):
        po = dict(po); unz = dict(po["unnormalizer_processor"])
        unz["stats"], d = patch(unz.get("stats"))
        po["unnormalizer_processor"] = unz; kwargs["postprocessor_overrides"] = po
        touched += [f"unnormalizer_processor:{x}" for x in d]
    return touched


def make_pre_post_processors(*a, **k):
    # PI05_FORCE_NORM が無ければ素通り（従来の挙動）。
    # 正規経路は `work/data/make_t2_stats_from_f1.py --write` で meta/stats.json を直す方。
    # こちらはデータセットが読み取り専用などで書けない時の代替で、効いたかは PI05_EXPECT_NORM が門にする。
    if _FORCE_PATH:
        try:
            import hashlib
            touched = _force_stats_into(k, _load_expect(_FORCE_PATH))
            INFO["force_norm"] = {"path": os.path.abspath(_FORCE_PATH),
                                  "sha256": hashlib.sha256(open(_FORCE_PATH, "rb").read()).hexdigest(),
                                  "patched": touched}
            print("WRAP force_norm", INFO["force_norm"])
        except Exception as e:  # noqa: BLE001
            INFO["force_norm_error"] = str(e); print("WRAP force_norm error", e)
            dump()
            raise RuntimeError(f"WRAP force_norm FAIL: {e}") from e
        dump()
    pre, post = _make_pp(*a, **k)
    try:
        ds_stats = k.get("dataset_stats")
        ov = k.get("preprocessor_overrides") or {}
        if ds_stats is None and isinstance(ov, dict):
            ds_stats = (ov.get("normalizer_processor") or {}).get("stats")
        st = _find_normalizer(pre)
        if st is None:
            INFO["normalizer"] = {"ok": False, "problems": ["preprocessor に normalizer step が見つからない"], "steps": [type(x).__name__ for x in (getattr(pre, "steps", None) or [])]}
        else:
            INFO["normalizer"] = _norm_record(st, ds_stats)
        stp = _find_normalizer(post)
        INFO["unnormalizer"] = ({"class": type(stp).__name__, "has_stats": bool(getattr(stp, "stats", None))} if stp is not None else {"present": False})
        print("WRAP normalizer", json.dumps({k2: v for k2, v in INFO["normalizer"].items() if k2 != "features"}), "| state q01/q99:",
              INFO["normalizer"].get("features", {}).get("observation.state", {}).get("q01"), INFO["normalizer"].get("features", {}).get("observation.state", {}).get("q99"))
        dump()
    except Exception as e:  # noqa: BLE001
        INFO["normalizer_error"] = str(e); print("WRAP normalizer error", e)
    # PI05_EXPECT_NORM が無ければここは丸ごと素通り（従来の挙動）
    if _EXPECT_PATH:
        halt = None
        try:
            import hashlib
            expect = _load_expect(_EXPECT_PATH)
            rec = {"path": os.path.abspath(_EXPECT_PATH),
                   "sha256": hashlib.sha256(open(_EXPECT_PATH, "rb").read()).hexdigest(),
                   "keys": sorted(expect.keys()), "strict": _EXPECT_STRICT}
            rec["normalizer"] = _expect_compare(_find_normalizer(pre), expect, "normalizer")
            rec["unnormalizer"] = _expect_compare(_find_normalizer(post), expect, "unnormalizer")
            rec["match"] = bool(rec["normalizer"]["match"] and rec["unnormalizer"]["match"])
            INFO["expect_norm"] = rec
            print(f"WRAP expect_norm match={rec['match']} strict={_EXPECT_STRICT} "
                  f"checked={rec['normalizer']['checked']}/{rec['unnormalizer']['checked']} src={rec['path']}")
            for tag in ("normalizer", "unnormalizer"):
                for p in rec[tag]["problems"]:
                    print("WRAP expect_norm NG:", p)
            if not rec["match"]:
                INFO.setdefault("normalizer", {})["ok"] = False
                halt = "正規化が PI05_EXPECT_NORM と一致しない（単因子条件が崩れる）"
        except Exception as e:  # noqa: BLE001
            INFO["expect_norm_error"] = str(e)
            print("WRAP expect_norm error", e)
            halt = f"PI05_EXPECT_NORM の照合自体に失敗した: {e}"
        dump()  # 落とす前に必ず記録を残す
        if halt is not None:
            if _EXPECT_STRICT:
                raise RuntimeError(f"WRAP expect_norm FAIL: {halt}（PI05_EXPECT_NORM_STRICT=0 で警告に落とせる）")
            print(f"WRAP expect_norm WARN（strict=0 なので続行）: {halt}")
    return pre, post
_pf.make_pre_post_processors = make_pre_post_processors

import lerobot.optim.factory as _of  # noqa: E402
_mk_opt = _of.make_optimizer_and_scheduler
def make_optimizer_and_scheduler(*a, **k):
    opt, sch = _mk_opt(*a, **k)
    if globals().get("_AUD") is not None:   # 実 update を torch optimizer の step 後フックで数える（train_audit）
        _AUD.attach_optimizer(opt)
    # T3-2: PEFT（LoRA）は make_policy の後に装着されるので、ここで policy を再計数する（Codex 2026-09-07 問 1）
    try:
        pol = a[1] if len(a) > 1 else k.get("policy")
        from collections import Counter
        n_all = n_tr = 0; dt_tr = Counter(); lora_names = 0
        for n, p in pol.named_parameters():
            n_all += p.numel()
            if p.requires_grad:
                n_tr += p.numel(); dt_tr[str(p.dtype)] += p.numel()
                if "lora" in n.lower(): lora_names += p.numel()
        peft_attached = bool(getattr(pol, "peft_config", None) is not None or hasattr(pol, "base_model") and "peft" in type(getattr(pol, "base_model", None)).__module__.lower() or lora_names > 0)
        INFO["after_peft"] = {"params_total": n_all, "trainable_params": n_tr, "trainable_dtypes": dict(dt_tr), "lora_param_numel": lora_names,
                              "peft_attached": peft_attached, "lora_ok": bool(0 < n_tr < n_all and lora_names > 0)}
        cfg0 = a[0] if a else k.get("cfg")
        INFO["drop_n_last_frames"] = getattr(getattr(cfg0, "policy", None), "drop_n_last_frames", None)
        print(f"WRAP after_peft total={n_all} trainable={n_tr} lora_numel={lora_names} peft_attached={peft_attached} drop_n_last_frames={INFO['drop_n_last_frames']}")
        dump()
    except Exception as e:  # noqa: BLE001
        INFO["after_peft_error"] = str(e); print("WRAP after_peft error", e)
    try:
        from collections import Counter
        c = Counter()
        for g in opt.param_groups:
            for p in g["params"]:
                c[str(p.dtype)] += p.numel()
        g0 = opt.param_groups[0]
        INFO["optimizer"] = {"type": type(opt).__name__, "param_dtypes": dict(c), "lr": g0.get("lr"), "weight_decay": g0.get("weight_decay"),
                             "betas": g0.get("betas"), "eps": g0.get("eps"), "foreach": g0.get("foreach"), "fused": g0.get("fused")}
        print("WRAP optimizer", INFO["optimizer"])
        dump()
    except Exception as e:  # noqa: BLE001
        INFO["opt_stats_error"] = str(e)
    # F1: scheduler の解決済み設定と、主要 step での LR（decay 後が floor になっているか）を記録する
    try:
        import hashlib, inspect
        cfg = a[0] if a else k.get("cfg")
        sc = getattr(cfg, "scheduler", None); steps = int(getattr(cfg, "steps", 0) or 0)
        rec = {"type": type(sc).__name__, "num_training_steps": steps}
        for f in ("num_warmup_steps", "num_decay_steps", "peak_lr", "decay_lr"):
            if hasattr(sc, f):
                rec[f] = getattr(sc, f)
        if sch is not None and hasattr(sch, "lr_lambdas") and "peak_lr" in rec:
            lam = sch.lr_lambdas[0]; pk = float(rec["peak_lr"]); w = int(rec.get("num_warmup_steps", 0)); dcy = int(rec.get("num_decay_steps", 0))
            pts = sorted({0, max(w - 1, 0), w, w + 1, dcy // 2, max(dcy - 1, 0), dcy, dcy + 1, steps})
            rec["lr_at"] = {str(s): float(lam(s)) * pk for s in pts}
        src = inspect.getsource(type(sc).build) if sc is not None else ""
        rec["build_source_sha256_16"] = hashlib.sha256(src.encode()).hexdigest()[:16]
        rec["auto_scale_in_source"] = "num_training_steps < self.num_decay_steps" in src
        INFO["scheduler"] = rec
        print("WRAP scheduler", rec)
        dump()
    except Exception as e:  # noqa: BLE001
        INFO["scheduler_error"] = str(e)
    return opt, sch
_of.make_optimizer_and_scheduler = make_optimizer_and_scheduler

# --- T3-3: 末尾 padding 行動を損失から除外する（PI05_MASK_ACTION_PAD=1 のとき）---
#   lerobot 0.4.4 の PI05Policy.forward は losses.mean() を取るだけで action_is_pad を見ない
#   （配布パッチが drop_n_last_frames=49 を入れているのはそのため）。drop を 0 に戻すには
#   ここでマスクを掛ける必要がある。実装は 2 段: Policy.forward で mask を stash し、
#   内側の PI05Pytorch.forward が返す (B, T, D) の損失に掛けて **有効要素だけで平均される**よう再正規化する。
if os.environ.get("PI05_MASK_ACTION_PAD") == "1":
    try:
        import lerobot.policies.pi05.modeling_pi05 as _mp05  # noqa: E402
        _PAD = {"cur": None}
        _MI = INFO.setdefault("pad_mask", {"enabled": True, "batches": 0, "missing_key": 0,
                                           "frac_padded_sum": 0.0, "frac_padded_mean": None, "applied": 0})
        _orig_pol_fwd = _mp05.PI05Policy.forward
        def _pol_fwd(self, batch, *a, **k):
            pad = batch.get("action_is_pad") if isinstance(batch, dict) else None
            _PAD["cur"] = pad
            _MI["batches"] += 1
            if pad is None:
                _MI["missing_key"] += 1
            return _orig_pol_fwd(self, batch, *a, **k)
        _mp05.PI05Policy.forward = _pol_fwd

        _orig_mdl_fwd = _mp05.PI05Pytorch.forward
        def _mdl_fwd(self, *a, **k):
            losses = _orig_mdl_fwd(self, *a, **k)
            pad = _PAD.get("cur")
            if pad is None or not torch.is_tensor(losses) or losses.dim() != 3:
                return losses
            m = (~pad.to(torch.bool)).to(losses.dtype).to(losses.device)      # (B, T) 有効=1
            if m.shape[:2] != losses.shape[:2]:
                return losses
            n_valid = m.sum()
            if float(n_valid) <= 0:
                return losses
            scale = float(m.numel()) / float(n_valid)   # 外側の losses.mean() が有効要素平均になるよう補正
            _MI["applied"] += 1
            _MI["frac_padded_sum"] += float(1.0 - n_valid / m.numel())
            _MI["frac_padded_mean"] = _MI["frac_padded_sum"] / max(_MI["applied"], 1)
            return losses * m.unsqueeze(-1) * scale
        _mp05.PI05Pytorch.forward = _mdl_fwd
        print("WRAP pad_mask ON（PI05Policy.forward / PI05Pytorch.forward を包んだ）")
    except Exception as e:  # noqa: BLE001
        INFO["pad_mask"] = {"enabled": False, "error": str(e)}
        print("WRAP pad_mask 失敗", e)
else:
    INFO["pad_mask"] = {"enabled": False}

# --- T3-3: 勾配クリッピングが同期時のみ呼ばれるかを記録（挙動は変えない）---
try:
    from accelerate import Accelerator as _Acc  # noqa: E402
    if hasattr(_Acc, "clip_grad_norm_"):
        _orig_clip = _Acc.clip_grad_norm_
        _CI = INFO.setdefault("grad_clip", {"calls": 0, "calls_when_not_sync": 0, "sync_unknown": 0})
        # 2026-09-23 B3: lerobot（grad-accum パッチ後）は毎 micro-step で clip_grad_norm_ を呼ぶ。GA>1 だと同期前の途中の勾配
        # （前半の micro-batch だけ）が先に閾値で切られうる（accelerate の説明例は sync_gradients のときだけ clip）。
        # PI05_CLIP_ONLY_ON_SYNC=1 なら同期しない micro-step では閾値を無限大にして**ノルムの計算だけ**する（勾配は変えない）。
        # **既定（未設定）は従来どおり**（B4 など他の台本は未設定のまま）。
        _CLIP_SYNC_ONLY = os.environ.get("PI05_CLIP_ONLY_ON_SYNC") == "1"
        # 2026-09-24 Codex 裁定（CODEX_RULING_EXEC_CONTROL_V4 裁定 1）: B3 Full は ON。train_audit の early_check が
        # 「sync_unknown=0・同期前の呼び出しが全件スキップ・同期時の呼び出し回数＝実 update 数」を照合する（hook_installed も見る）。
        _CI["only_on_sync"] = _CLIP_SYNC_ONLY
        _CI["hook_installed"] = True
        _CI.setdefault("skipped_not_sync", 0)
        _CI.setdefault("calls_when_sync", 0)
        def _clip(self, *a, **k):
            _CI["calls"] += 1
            sg = getattr(self, "sync_gradients", None)
            if sg is None:
                _CI["sync_unknown"] += 1
            elif sg:
                _CI["calls_when_sync"] += 1
            else:
                _CI["calls_when_not_sync"] += 1
                if _CLIP_SYNC_ONLY:
                    _CI["skipped_not_sync"] += 1
                    if len(a) >= 2:
                        a = (a[0], float("inf")) + tuple(a[2:])
                    else:
                        k = dict(k); k["max_norm"] = float("inf")
            return _orig_clip(self, *a, **k)
        _Acc.clip_grad_norm_ = _clip
except Exception as e:  # noqa: BLE001
    INFO["grad_clip_hook_error"] = str(e)

# --- step timing hook（update_policy を包む） ---
import lerobot.scripts.lerobot_train as _lt  # noqa: E402
# rft_dataset installs first and imports lerobot_train before this wrapper.
# Its `from ... import ...` aliases then retain the old factory functions.
# Bind all three aliases: processor stats, parent-weight identity, and the
# optimizer step post-hook used by TrainAudit's early_check.
_lt.make_policy = make_policy
_lt.make_pre_post_processors = make_pre_post_processors
_lt.make_optimizer_and_scheduler = make_optimizer_and_scheduler
INFO["factory_bindings"] = {
    "policy": _lt.make_policy is make_policy,
    "processors": _lt.make_pre_post_processors is make_pre_post_processors,
    "optimizer": _lt.make_optimizer_and_scheduler is make_optimizer_and_scheduler,
}
if not all(INFO["factory_bindings"].values()):
    raise RuntimeError(f"WRAP factory alias binding failed: {INFO['factory_bindings']}")
dump()

# --- [2026-09-09 追加] EpisodeAwareSampler の global/local 添字不整合を直す -----------------
# 症状: `--dataset.episodes` で部分集合にすると 1 step 目で
#       IndexError: Invalid key: 2025958 is out of bounds for size 811202
# 原因: lerobot_train.py が EpisodeAwareSampler へ meta の **global** フレーム番号
#       （dataset_from_index / dataset_to_index、0..total_frames-1）を渡すのに、
#       LeRobotDataset.load_hf_dataset() は episodes で絞った行だけを読み **0 起点に振り直す**。
#       drop_n_last_frames が policy config にある時だけこの経路に入るため、
#       全データ学習（global==local）や、この属性を持たない policy では露見しなかった。
# 方針: episode_indices_to_use が None のとき（= 全データ）は **1 ビットも変えず**素通し。
#       部分集合のときだけ、選抜 episode を昇順に詰め直した local 添字で組み直す。
#       サンプル本数は現行と同一で、**学習する内容は変わらない**（Mac 実データで検証: 576,002 本、
#       最大添字 2,234,464 → 811,152、行数 811,202。work/reports/T2F1_SAMPLER_FIX_20260909.md）。
# 前提: 絞った hf_dataset が episode_index 昇順に並ぶこと。Mac の libero_plus（instance と
#       stats.json の sha256 が一致 = バイト同一）で 4,800/4,800 本が昇順・行数一致を確認済み。
#       前提が崩れたら黙って進まないよう、下で local 合計と選抜フレーム総数の一致を検査する。
import lerobot.datasets.sampler as _smp  # noqa: E402

_OrigEAS = _smp.EpisodeAwareSampler


class _EASLocalIndex(_OrigEAS):
    def __init__(self, dataset_from_indices, dataset_to_indices, episode_indices_to_use=None,
                 drop_n_first_frames=0, drop_n_last_frames=0, shuffle=False):
        if episode_indices_to_use is None:
            super().__init__(dataset_from_indices, dataset_to_indices, None,
                             drop_n_first_frames, drop_n_last_frames, shuffle)
            INFO["sampler_patch"] = {"applied": False, "reason": "full dataset (global==local)",
                                     "n_indices": len(self.indices)}
            return
        use = {int(x) for x in episode_indices_to_use}
        lens = [int(e) - int(s) for s, e in zip(dataset_from_indices, dataset_to_indices)]
        indices, local = [], 0
        for ep, n in enumerate(lens):
            if ep in use:
                lo, hi = local + drop_n_first_frames, local + n - drop_n_last_frames
                if hi > lo:
                    indices.extend(range(lo, hi))
                local += n
        sel_frames = sum(lens[e] for e in use if 0 <= e < len(lens))
        if local != sel_frames:
            raise RuntimeError(f"sampler patch: local 合計 {local} != 選抜フレーム総数 {sel_frames}")
        if indices and max(indices) >= local:
            raise RuntimeError(f"sampler patch: 最大添字 {max(indices)} が行数 {local} を超えた")
        self.indices = indices
        self.shuffle = shuffle
        INFO["sampler_patch"] = {
            "applied": True, "n_episodes_used": len(use), "n_rows_local": local,
            "n_indices": len(indices), "max_index": (max(indices) if indices else None),
            "drop_n_first_frames": drop_n_first_frames, "drop_n_last_frames": drop_n_last_frames,
        }
        print(f"WRAP sampler_patch applied: episodes={len(use)} rows={local} "
              f"indices={len(indices)} max_index={max(indices) if indices else None}", flush=True)


# lerobot_train は `from ... import EpisodeAwareSampler` で束縛済みなので、**両方**差し替える
_smp.EpisodeAwareSampler = _EASLocalIndex
_lt.EpisodeAwareSampler = _EASLocalIndex
INFO["sampler_patch_installed"] = True
# ------------------------------------------------------------------------------------------
# --- [2026-09-11 追加] 学習画像を評価入力に近づける読み込み時変換（T2F1C。PI05_TRAIN_IMG_EVAL_LIKE=1 のときだけ）------
# 正本: work/plans/PREREG_T2_F1C_IMGEVAL_20260912.md / work/tools/pi05_imgeval.py。
# 方針: 環境変数が無ければ **1 バイトも変えず**素通し（F0/F1/T2F1/T2F1B の走りは変わらない）。
#       有効時は lerobot_train.make_dataset を包み、構築直後に dataset.image_transforms へ EvalLikeResize を差す。
#       ★ 差しただけで「適用済み」とは判定しない: 同じ index を変換なし／ありで取り出し、実際に学習へ渡る front / wrist が
#         「単回適用の参照」と一致し「二重適用」と不一致であることを、最初の学習更新の前に確認する（不一致なら学習を始めない）。
#       検査は dataset[idx] を 2 回呼ぶだけで、乱数列（torch RNG）もサンプラ順も触らない。
IMG_EVAL_LIKE = os.environ.get("PI05_TRAIN_IMG_EVAL_LIKE") == "1"
INFO["img_eval_like"] = {"enabled": IMG_EVAL_LIKE, "applied": False}
if IMG_EVAL_LIKE:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pi05_imgeval import EvalLikeResize, probe_sha256, selfcheck as _img_selfcheck  # noqa: E402

    _orig_make_dataset = _lt.make_dataset

    def make_dataset(cfg, *a, **k):
        ds = _orig_make_dataset(cfg, *a, **k)
        info = INFO["img_eval_like"]
        if getattr(ds, "image_transforms", None) is not None:
            raise SystemExit("IMG_EVAL_LIKE: 既存の image_transforms が有効（二重適用の恐れ）。停止")
        chk = _img_selfcheck(verbose=False)
        info["selfcheck"] = chk
        if not chk["ok"]:
            raise SystemExit(f"IMG_EVAL_LIKE: selfcheck FAIL {chk}")
        tr = EvalLikeResize()
        cams = list(ds.meta.camera_keys)
        idx = 0
        ds.image_transforms = None
        raw = ds[idx]
        ds.image_transforms = tr
        got = ds[idx]
        detail = {}
        for cam in cams:
            ref1 = tr(raw[cam])
            ref2 = tr(ref1)
            same1 = bool(torch.equal(got[cam], ref1))
            same2 = bool(torch.equal(got[cam], ref2))
            detail[cam] = {"shape": list(got[cam].shape), "dtype": str(got[cam].dtype),
                           "equal_single_apply": same1, "equal_double_apply": same2,
                           "max_abs_change_vs_raw_levels": float((got[cam] - raw[cam]).abs().max() * 255)}
            if not same1 or same2:
                raise SystemExit(f"IMG_EVAL_LIKE: 実適用の検査に失敗 {cam}: {detail[cam]}")
        info.update({"applied": True, "verified_index": idx, "cameras": cams, "detail": detail,
                     "transform": repr(tr), "probe_sha256": probe_sha256(), "video_backend": getattr(ds, "video_backend", None)})
        print(f"WRAP img_eval_like applied+verified: cams={cams} probe={info['probe_sha256'][:16]} backend={info['video_backend']}", flush=True)
        dump()
        return ds

    _lt.make_dataset = make_dataset
    INFO["img_eval_like"]["hook_installed"] = True
# ------------------------------------------------------------------------------------------
_upd = _lt.update_policy
_T = {"n": 0, "t_start_steady": None, "t_last": None}
def update_policy(*a, **k):
    r = _upd(*a, **k)
    _T["n"] += 1; now = time.time()
    if _T["n"] == 10:
        _T["t_start_steady"] = now
    _T["t_last"] = now
    return r
_lt.update_policy = update_policy

def _atexit():
    try:
        if torch.cuda.is_available():
            INFO["peak_reserved_MiB"] = int(torch.cuda.max_memory_reserved() / 1048576)
            INFO["peak_allocated_MiB"] = int(torch.cuda.max_memory_allocated() / 1048576)
            INFO["gpu_total_MiB"] = int(torch.cuda.get_device_properties(0).total_memory / 1048576)
        if _T["t_start_steady"] and _T["n"] > 10:
            INFO["s_per_step_steady"] = round((_T["t_last"] - _T["t_start_steady"]) / (_T["n"] - 10), 4)
        INFO["steps_done"] = _T["n"]
        print(f"WRAP peak_reserved_MiB={INFO.get('peak_reserved_MiB')} peak_allocated_MiB={INFO.get('peak_allocated_MiB')} s_per_step_steady={INFO.get('s_per_step_steady')} steps={_T['n']}")
    finally:
        dump()
atexit.register(_atexit)

# --- draw manifest（Codex 2026-09-23 裁定 §2。B3 で追加）--------------------------------------
# PI05_DRAW_MANIFEST があれば、EpisodeAwareSampler の順序を凍結した global index の列へ固定する。
# 既定の順序は torch.randperm（global RNG）で決まり、PEFT 装着や microbatch 数で Full と LoRA がずれるため。
# **環境変数を渡さなければ何も変わらない**（F0/F1/T2F1/T3 系の走りには影響しない）。
_DM = os.environ.get("PI05_DRAW_MANIFEST")
if _DM:
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in (_here, os.path.join(_here, "b3")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from draw_manifest_sampler import install as _install_dm  # noqa: E402
    from lerobot.datasets import sampler as _smp  # noqa: E402
    with open(_DM) as _f:
        _dmd = json.load(_f)
    # B3 は全量なので global＝local（`global_indices`）。B4 は episode 部分集合で _EASLocalIndex の local 空間（`sampler_indices`）
    _dm_idx = _dmd["sampler_indices"] if "sampler_indices" in _dmd else _dmd["global_indices"]
    INFO["draw_manifest"] = _install_dm(
        _smp.EpisodeAwareSampler, _dm_idx,
        record={"path": _DM, "name": _dmd.get("name"), "n_draws": _dmd.get("n_draws"),
                "effective_batch": _dmd.get("effective_batch")},
        on_event=dump)   # sampler 構築・走り始め・走り終わりで wrap JSON を書く（起動前ゲートが「フックが効いた」を見られるように）
    print(f"WRAP draw_manifest {_dmd.get('name')} n={len(_dm_idx)} space={'local' if 'sampler_indices' in _dmd else 'global'} (sampler 順序を固定)")
    dump()

# --- 学習の実績の監査（Codex 2026-09-23 レビュー §3 A・D。PI05_TRAIN_AUDIT=1 のときだけ）-----------------------------
# 実 update 数（optimizer の step 後フック）・学習した sample（batch の index）・最初の K update の照合（GA・実効 batch・
# manifest 先頭との一致・PAD）・checkpoint の保存完了（save_checkpoint が返った後に READY.json）。**渡さなければ何も変わらない。**
_AUD = None
if os.environ.get("PI05_TRAIN_AUDIT") == "1":
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in (_here, os.path.join(_here, "b3")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from train_audit import TrainAudit, expected_from_manifest  # noqa: E402
    _exp = expected_from_manifest(_dmd) if _DM else None
    _AUD = TrainAudit(INFO, dump, expected=_exp,
                      expect_ga=int(os.environ["PI05_EXPECT_GA"]) if os.environ.get("PI05_EXPECT_GA") else None,
                      expect_eff=int(os.environ["PI05_EXPECT_EFF"]) if os.environ.get("PI05_EXPECT_EFF") else None,
                      early_k=int(os.environ.get("PI05_EARLY_UPDATES", "3")),
                      check_pad=os.environ.get("PI05_MASK_ACTION_PAD") == "1", pad_info=INFO.get("pad_mask"),
                      check_clip_sync=os.environ.get("PI05_CLIP_ONLY_ON_SYNC") == "1")
    _AUD.install(_lt)
    print(f"WRAP train_audit installed ga={_AUD.ga} eff={_AUD.eff} k={_AUD.k} pad={_AUD.check_pad} clip_sync={_AUD.check_clip_sync} expected={len(_exp) if _exp else None}", flush=True)
    dump()

if __name__ == "__main__":
    sys.argv = ["lerobot-train", *sys.argv[1:]]
    _lt.main()
