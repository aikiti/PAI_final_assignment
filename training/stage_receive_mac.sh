#!/usr/bin/env bash
# =============================================================================
# 官給 instance から download した staging 1 label を Mac で受け入れる（B3・B4 共用。2026-09-23）
#   Codex レビュー §3 C: 「Mac 側 SHA 照合完了」を記録し、instance の RECEIVED と結ぶ。
#
#   使い方: bash work/tools/stage_receive_mac.sh <download したフォルダ> <label>
#     フォルダに要る物: <label>.tar.part_*（全片）・<label>.sha256・<label>.STAGED
#   やること:
#     1) <label>.sha256 で全片を照合（1 片でも欠け・違いがあれば止める）
#     2) 片の合計の大きさが STAGED の記録と一致
#     3) 片を連結して展開 → <フォルダ>/<label>/ に export（Full は model.safetensors・config.json、LoRA は pretrained_model/adapter_model.safetensors）
#     4) instance の Terminal に貼る 1 行を出す:  bash ~/<台本> rec done <RUN_ID> <label> <sha256(<label>.sha256)>
#        （instance 側は自分の sha 一覧と照合してから RECEIVED にする。違えば RECEIVED にならない）
#   ★ 分割片はこのスクリプトでは消さない（消すのは本人 GO の後。消去は今回追加した物だけ）。
# =============================================================================
set -u -o pipefail
D="${1:-}"; LAB="${2:-}"
[ -n "$D" ] && [ -n "$LAB" ] || { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
cd "$D" || exit 2
[ -f "${LAB}.sha256" ] && [ -f "${LAB}.STAGED" ] || { echo "★ ${LAB}.sha256 と ${LAB}.STAGED を一緒に download する"; exit 2; }
if command -v sha256sum >/dev/null 2>&1; then SHA() { sha256sum "$@"; }; CHECK() { sha256sum -c "$1" --quiet; }
else SHA() { shasum -a 256 "$@"; }; CHECK() { shasum -a 256 -c "$1" --quiet; }; fi
n_list=$(wc -l < "${LAB}.sha256" | tr -d ' '); n_have=$(ls "${LAB}".tar.part_* 2>/dev/null | wc -l | tr -d ' ')
[ "$n_list" = "$n_have" ] || { echo "★ 片の数が違う（一覧 ${n_list}・手元 ${n_have}）。download が途中"; exit 1; }
CHECK "${LAB}.sha256" || { echo "★ sha 照合に失敗（download が壊れている）。RECEIVED にしない"; exit 1; }
want_bytes=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['bytes'])" "${LAB}.STAGED") || exit 1
got_bytes=$(cat "${LAB}".tar.part_* | wc -c | tr -d ' ')
[ "$want_bytes" = "$got_bytes" ] || { echo "★ 大きさが STAGED の記録と違う（${got_bytes} ≠ ${want_bytes}）"; exit 1; }
[ -e "$LAB" ] && { echo "★ 展開先 ${D}/${LAB} が既にある（上書きしない）"; exit 1; }
cat "${LAB}".tar.part_* | tar -xf - || { echo "★ 展開に失敗"; exit 1; }
if [ -s "${LAB}/model.safetensors" ] && [ -s "${LAB}/config.json" ]; then kind="Full export"
elif [ -s "${LAB}/pretrained_model/adapter_model.safetensors" ]; then kind="LoRA adapter"
else echo "★ 展開した中に重みが無い"; exit 1; fi
rid=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['run_id'])" "${LAB}.STAGED")
scr=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['script'])" "${LAB}.STAGED")
val=$(SHA "${LAB}.sha256" | cut -d' ' -f1)
echo "MAC_RECEIVE_OK ${LAB}（${kind}、${n_have} 片・${got_bytes} B）→ ${D}/${LAB}/"
echo "instance の Terminal に貼る:"
echo "bash ~/${scr} rec done ${rid} ${LAB} ${val}"
