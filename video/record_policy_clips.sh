#!/usr/bin/env bash
# =============================================================================
# 講座「Physical AI 応用1」最終課題: 学習前・学習後のモデルを Mac で動かし、本の場面の動画を取る（重い工程・約 25 分）。
#   学習前 = M-DEMO-1 の親（b3_F_003000）・env seed 42
#   学習後 = M-DEMO-1 の M1（md_M1_001000）・env seed 43（本の最初の回が成功でなければ seed 42 でも取る）
#   M-DEMO-1 の評価台本 work/eval/mdemo1/run_mdemo_panel.sh を試運転モード（MDEMO_OUT=…/dryrun_course_video_*）で呼ぶ。
#   本番の runs/ には書かない。録画は配布 kit の T3_VIDEO_DIR（各 task の episode 0 だけ）。GPU 0 分・LB 枠 0。
# 条件: AC 電源・蓋は開けたまま・本選の Mac 評価/学習/手動収録が動いていないこと。
#   起動前に自動で確かめ、1 つでも違えば何もせずに止まる。
# 使い方: bash work/tools/course_final/record_policy_clips.sh
# 出力: work/reports/course_final_20261009/raw/<model>_s<seed>/*.mp4 と raw/CLIPS.sha256。
#   録画済み（.done あり）のモデルは再実行しても飛ばす。
# =============================================================================
set -u -o pipefail
REPO=/Users/USERNAME/PhysicalAI/PARC2026
cd "${REPO}" || exit 1
CF="${REPO}/work/reports/course_final_20261009"
RAW="${CF}/raw"
EVAL_OUT="${REPO}/work/reports/honsen2_mdemo1_eval_20260926"
PARENT_EXPORT="${PARENT_EXPORT:-/Users/USERNAME/Downloads/b3_F_003000}"
M1_EXPORT="${M1_EXPORT:-/Users/USERNAME/Downloads/md_M1_hf/md_M1_001000}"
# M-DEMO-1 の評価時の結び付け（runs/mdemo1_{parent,m1}.binding.json）の weights_sha256
PARENT_W_SHA=6ea165ebe3f4e7180bb9f0e7a5e244aaf0952f54a34bccffd4aeaa56ced78818
M1_W_SHA=d0a6cd8635583ff437023e8b17d7dd49257581f63c5c4d5318f9db5d5bd31ae6
SERVER_ZIP=submissions/pi05_t2f1_t2f1_005000_nb_lic.zip
BOOK=take_the_yellow_book_off_the_shelf_and_put_it_on_the_table
mkdir -p "${RAW}" "${CF}/logs"
LOG="${CF}/logs/record_policy_clips_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== 講座課題の録画 $(date '+%F %T') ==="
die() { echo "★ $1"; exit "${2:-1}"; }

# ---- 起動前の確認（どれか 1 つでも違えば何もしない）
pmset -g ps | head -1 | grep -q "AC Power" || die "AC 電源でない。つないで、蓋を開けたまま起動する。" 5
# 評価の連鎖は腕と腕の間でサーバを止めるので、サーバだけでなく work/eval/ の台本・期限の見張りも見る（10/9 の H_MIXB 連鎖で確認）
BUSY=$(ps -Ao pid,command | grep -E "work/eval/|run_panel|run_selfeval|evaluate\.py|serve_qself|policy_server|lerobot_train|pi05_train_wrapper|train_lora|teleop_record|deadline_guard" | grep -v grep)
if [ -n "${BUSY}" ]; then
  echo "${BUSY}"
  die "本選の評価・学習・手動収録が動いている。重ねないため中止（終わってから起動する）。" 6
fi
curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && die "port 8000 が使用中。本選の推論サーバと重ねないため中止。" 6
FREE_GB=$(df -g "${REPO}" | awk 'NR==2 {print $4}')
[ "${FREE_GB:-0}" -ge 10 ] || die "空き容量が ${FREE_GB} GB しかない（10 GB 以上で起動する）。" 7
[ -s "${SERVER_ZIP}" ] || die "推論サーバを取り出す zip が無い: ${SERVER_ZIP}" 7
for pair in "parent|${PARENT_EXPORT}|${PARENT_W_SHA}" "m1|${M1_EXPORT}|${M1_W_SHA}"; do
  IFS='|' read -r m e s <<< "${pair}"
  [ -s "${e}/model.safetensors" ] || die "${m} の重みが無い: ${e}（消していないか確かめる）" 7
  echo "（${m} の重みの sha を照合中…）"
  got=$(shasum -a 256 "${e}/model.safetensors" | cut -d' ' -f1)
  [ "${got}" = "${s}" ] || die "${m} の重みの sha が M-DEMO-1 の評価時と違う: ${got}" 7
done
python3 work/eval/mdemo1/mdemo_pins.py check-code || die "評価コードが M-DEMO-1 の凍結値と違う（10/2 の結果と同じ条件で録れない）。" 7
echo "起動前の確認は全部通過 $(date '+%T')"

# ---- 1 モデル分の録画（試運転の出力先は毎回新しく作る。本番の runs/ に書かない）
record() {  # $1=model $2=seed $3=export
  local model="$1" seed="$2" export_dir="$3"
  local vdir="${RAW}/${model}_s${seed}"
  if [ -f "${vdir}/.done" ]; then echo "SKIP: ${model}（seed ${seed}）は録画済み: ${vdir}"; return 0; fi
  if [ -d "${vdir}" ]; then mv "${vdir}" "${vdir}.prev_$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "${vdir}"
  local dry="${EVAL_OUT}/dryrun_course_video_${model}_s${seed}_$(date +%Y%m%d_%H%M%S)"
  echo "--- ${model}（seed ${seed}）を録画 $(date '+%T')  試運転の出力: ${dry}"
  T3_VIDEO_DIR="${vdir}" ENV_SEED="${seed}" MDEMO_OUT="${dry}" \
    caffeinate -dims bash work/eval/mdemo1/run_mdemo_panel.sh "${model}" main "${export_dir}"
  local rc=$?
  echo "評価台本の終了コード ${rc}（最後の表の判定で非 0 になることがある。録画が取れているかで判断する）"
  ls -la "${vdir}"
  ls "${vdir}/${BOOK}"__*.mp4 >/dev/null 2>&1 || { echo "★ ${model} の本の動画が無い"; return 1; }
  python3 work/tools/course_final/summarize_dryrun.py "${dry}" || true
  touch "${vdir}/.done"
}

record parent 42 "${PARENT_EXPORT}" || die "学習前の録画に失敗（ログ: ${LOG}）"
record m1 43 "${M1_EXPORT}" || die "学習後の録画に失敗（ログ: ${LOG}）"
if [ ! -f "${RAW}/m1_s43/${BOOK}__SUCCESS.mp4" ]; then
  echo "学習後（seed 43）の本の最初の回が成功でなかった。seed 42 でも取る（10/2 の記録では seed 42 の最初の回も成功）"
  record m1 42 "${M1_EXPORT}" || die "学習後（seed 42）の録画に失敗（ログ: ${LOG}）"
fi
( cd "${RAW}" && shasum -a 256 ./*_s*/*.mp4 > CLIPS.sha256 )
echo "=== 録画 DONE $(date '+%F %T') ==="
cat "${RAW}/CLIPS.sha256"
