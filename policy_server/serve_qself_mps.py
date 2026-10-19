#!/usr/bin/env python3
"""Q-SELF-P 収録用: seed 差し込み済みの複製（policy_server_qself.py）を Mac(MPS) で起動する薄いラッパ。

`~/pi05_kit/serve_mps.py` と同じことをする（device だけ差し替えて同じ app を起動する）。違いは読むファイル名を
`SERVER_FILE` で指定できる点だけ。**出荷ファイルもこのラッパも提出物には入らない。**

環境変数: SUB（展開先）/ SERVER_FILE（既定 policy_server_qself.py）/ CKPT / DEVICE（既定 mps）/ PORT（既定 8000）
          QSELF_PROCESS_SEED（複製が読む。未設定なら出荷物と同じ 20260829）
"""
import importlib.util
import os
from pathlib import Path

SUB = Path(os.environ["SUB"])
SERVER_FILE = SUB / os.environ.get("SERVER_FILE", "policy_server_qself.py")
CKPT = Path(os.environ.get("CKPT", SUB / "model_weights"))
DEVICE = os.environ.get("DEVICE", "mps")
PORT = int(os.environ.get("PORT", "8000"))

os.environ["PI05_CKPT"] = str(CKPT)
spec = importlib.util.spec_from_file_location("_qself_server", SERVER_FILE)
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

print(f"[serve_qself] SERVER_FILE={SERVER_FILE}", flush=True)
print(f"[serve_qself] CKPT={CKPT}", flush=True)
print(f"[serve_qself] DEVICE={DEVICE} QSELF_PROCESS_SEED={os.environ.get('QSELF_PROCESS_SEED', '(unset)')}", flush=True)
srv.set_policy(srv.MyPolicy(device=DEVICE))
print(f"[serve_qself] listening on 127.0.0.1:{PORT}", flush=True)
import uvicorn

uvicorn.run(srv.app, host="127.0.0.1", port=PORT, log_level="warning")
