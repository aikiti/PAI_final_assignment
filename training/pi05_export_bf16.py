#!/usr/bin/env python3
"""F0 段 6 — Full FT checkpoint（pretrained_model）を推論用 bf16 に落として export し、検査する（v2、Codex 反映）。

- model.safetensors を bf16 に cast（fp32 master 学習なら 14.5GB → 7.23GB）。全要素の有限性を検査
- --reference（merged old9k）とキー集合・shape を照合（train-only の差は問題として列挙）
- config.json: dtype=bfloat16 / device=cuda / push_to_hub=false / pretrained_path=None、
  学習専用キー drop_n_last_frames / drop_n_first_frames を除去（merge_lora_pi05.py と同じ処理）
- policy_preprocessor / policy_postprocessor と、それらが参照する *_processor.safetensors を複写して存在確認
- --tokenizer-dir を tokenizer/ として同梱（checkpoint には無い）
- train_config.json / README.md は含めない。EXPORT_REPORT.json を書く。検査失敗は非 0 終了
- size は safetensors ヘッダ（metadata）の長さ差を許容（±4096 B）。F0 実測: merge 出力より 128 B 短い（metadata 差）
"""
import argparse, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

TRAIN_ONLY = ("drop_n_last_frames", "drop_n_first_frames")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-dir", default=None); ap.add_argument("--reference", default=None)
    ap.add_argument("--expect-size", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    problems = []
    src_w = os.path.join(a.src, "model.safetensors")
    tensors = {}; n_lora = 0; dt_in = {}
    with safe_open(src_w, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            dt_in[str(t.dtype)] = dt_in.get(str(t.dtype), 0) + 1
            if "lora_" in k:
                n_lora += 1
            if not torch.isfinite(t.to(torch.float32)).all():
                problems.append(f"non-finite: {k}")
            tensors[k] = t.to(torch.bfloat16).contiguous()
    if a.reference:
        with safe_open(a.reference, "pt") as r:
            rk = set(r.keys()); sk = set(tensors)
            if rk != sk:
                problems.append(f"key set differs from reference: only_ref={sorted(rk-sk)[:3]} only_src={sorted(sk-rk)[:3]}")
            for k in rk & sk:
                if tuple(r.get_slice(k).get_shape()) != tuple(tensors[k].shape):
                    problems.append(f"shape differs: {k}")
    out_w = os.path.join(a.out, "model.safetensors")
    save_file(tensors, out_w, metadata={"format": "pt"})
    size = os.path.getsize(out_w)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    changed = {}
    for k, v in (("dtype", "bfloat16"), ("device", "cuda"), ("push_to_hub", False), ("pretrained_path", None)):
        if cfg.get(k) != v:
            changed[k] = (cfg.get(k), v); cfg[k] = v
    removed = [k for k in TRAIN_ONLY if cfg.pop(k, None) is not None]
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    copied = []
    for fn in sorted(os.listdir(a.src)):
        if fn.startswith("policy_"):
            shutil.copy2(os.path.join(a.src, fn), os.path.join(a.out, fn)); copied.append(fn)
    for js in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = os.path.join(a.out, js)
        if not os.path.isfile(p):
            problems.append(f"missing {js}"); continue
        txt = open(p).read()
        # processor JSON が参照する state ファイル名（*_processor.safetensors）が存在するか
        import re
        for ref in set(re.findall(r'"([^"]+_processor\.safetensors)"', txt)):
            if not os.path.isfile(os.path.join(a.out, os.path.basename(ref))):
                problems.append(f"{js} references missing state file {ref}")
    if a.tokenizer_dir:
        td = os.path.join(a.out, "tokenizer"); shutil.rmtree(td, ignore_errors=True); shutil.copytree(a.tokenizer_dir, td)
        for fn in ("tokenizer.json", "tokenizer_config.json"):
            if not os.path.isfile(os.path.join(td, fn)):
                problems.append(f"tokenizer/{fn} missing")
    else:
        problems.append("tokenizer-dir not given")
    dt_out = {}
    with safe_open(out_w, "pt") as f:
        for k in f.keys():
            d = f.get_slice(k).get_dtype(); dt_out[d] = dt_out.get(d, 0) + 1
    rep = {"src": a.src, "n_tensors": len(tensors), "lora_keys": n_lora, "dtype_in": dt_in, "dtype_out": dt_out,
           "size_bytes": size, "expect_size": a.expect_size, "size_matches": (a.expect_size is None or abs(size - a.expect_size) <= 4096), "size_diff_bytes": (None if a.expect_size is None else size - a.expect_size),
           "config_changed": changed, "train_only_removed": removed, "copied": copied, "problems": problems}
    if n_lora:
        problems.append(f"lora keys present: {n_lora}")
    if set(dt_out) != {"BF16"}:
        problems.append(f"dtype_out not all BF16: {dt_out}")
    if not rep["size_matches"]:
        problems.append(f"size {size} != expect {a.expect_size}")
    rep["problems"] = problems
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(a.out, "EXPORT_REPORT.json"), "w"), indent=1)
    print("EXPORT_OK" if not problems else "EXPORT_CHECK_FAILED")
    return 0 if not problems else 1

if __name__ == "__main__":
    sys.exit(main())
