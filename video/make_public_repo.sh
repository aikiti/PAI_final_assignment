#!/usr/bin/env bash
# =============================================================================
# 講座「Physical AI 応用1」最終課題: GitHub に公開する「使用したコード一式」のフォルダを作る（モデルを動かさない・数秒）。
#   自作のコードだけを集め、公開に向かないものを除き、本名入りのパスを伏せる。
#   ・推論サーバ本体 policy_server.py（本選の配布リポジトリ〔招待制〕の例を改変したもの）と *.bak_* は入れない
#   ・Track3 の道具（人手デモの記録ツール・逆転タスクの生成・人手デモを混ぜた学習と評価の台本）は、
#     PARC2026 本選の期間中（11/24 まで）は入れない。入っていないことを出力前に検査する
#   ・作成者のユーザー名（Mac の絶対パスに入っている）を USERNAME に置き換える
#   ・gitleaks で秘密情報を、grep でユーザー名・メールアドレスが残っていないかを検査し、見つかれば出力しない
#   git の初期化・push はしない（GitHub への公開は本人の GO の後）。
# 使い方: bash work/tools/course_final/make_public_repo.sh
# 出力: PUBLIC_REPO_DIR（既定 ~/PhysicalAI/PAI_final_assignment）。作り直すときも .git は残す。
# =============================================================================
set -u -o pipefail
REPO="${HOME}/PhysicalAI/PARC2026"
cd "${REPO}" || exit 1
OUT="${PUBLIC_REPO_DIR:-${HOME}/PhysicalAI/PAI_final_assignment}"
die() { echo "★ $1"; exit "${2:-1}"; }
# 伏せる・検査する個人情報は実行時に Mac と git の設定から読む（このファイル自体も公開されるので、文字列で書かない）
NAME=$(id -un)
FULL_NAME=$(id -F 2>/dev/null || true)
EMAIL=$(git -C "${REPO}" config user.email 2>/dev/null || true)
[ -n "${NAME}" ] || die "ユーザー名を読めない"
STAGE_PARENT=$(mktemp -d "${TMPDIR:-/tmp}/course_public.XXXXXX") || die "一時フォルダを作れない"
trap 'rm -rf "${STAGE_PARENT}"' EXIT
S="${STAGE_PARENT}/PAI_final_assignment"
mkdir -p "${S}"
EXC=(--exclude '__pycache__' --exclude '.DS_Store' --exclude '*.pyc' --exclude '.pytest_cache' --exclude '*.bak_*'
     --exclude '*.npz' --exclude '*.safetensors' --exclude '*.mp4' --exclude 'PUBLIC_README.md')
cp_dir() { [ -d "$1" ] || die "フォルダが無い: $1"; mkdir -p "${S}/$2"; rsync -a "${EXC[@]}" "$1/" "${S}/$2/"; }
cp_file() { [ -f "$1" ] || die "ファイルが無い: $1"; mkdir -p "${S}/$2"; cp -p "$1" "${S}/$2/"; }

cp work/tools/course_final/PUBLIC_README.md "${S}/README.md" || die "README の原稿が無い"
for f in work/tools/pi05_train_wrapper.py work/tools/pi05_export_bf16.py work/tools/stage_receive_mac.sh \
         work/tools/b3/draw_manifest_sampler.py work/tools/b3/train_audit.py; do
  cp_file "${f}" training
done
# b3_panel からは評価の流れで使うものだけ（B3 の LoRA 統合用の b3_merge_prep.py 等は入れない）
for f in run_panel.py reset_contract.py tables.py binding.py mask_contract_sidecar.py; do
  cp_file "work/eval/b3_panel/${f}" evaluation/b3_panel
done
cp_file work/tools/qself/seed_patch_policy_server.py policy_server
cp_file work/tools/qself/serve_qself_mps.py policy_server
cp_dir work/tools/course_final video

# ---- 公開に向かないものを伏せる・除く
HITS=$(LC_ALL=C grep -rIl -- "${NAME}" "${S}" || true)
if [ -n "${HITS}" ]; then
  while IFS= read -r f; do
    LC_ALL=C sed -i '' "s/${NAME}/USERNAME/g" "${f}" || die "ユーザー名の置き換えに失敗: ${f}"
  done <<< "${HITS}"
fi
[ -z "$(find "${S}" -name 'policy_server.py')" ] || die "policy_server.py が入っている"
T3=$(find "${S}" \( -name 'teleop_record*' -o -name 'goal_overlay*' -o -name 'make_reverse_bddl*' -o -name 'pi05_mdemo1*' \
       -o -name 'pi05_t3_*' -o -name 'mdemo*' -o -name 'hm_train_entry*' -o -name 'families*.json' \))
[ -z "${T3}" ] || { echo "${T3}"; die "Track3 の道具が入っている（出力していない）"; }
[ -z "$(find "${S}" -name '*.bak_*')" ] || die "バックアップファイルが入っている"
for p in "${NAME}" "${FULL_NAME}" "${EMAIL}" "${EMAIL%%@*}"; do
  [ -n "${p}" ] || continue
  LEFT=$(grep -rIlF -- "${p}" "${S}" || true)
  [ -z "${LEFT}" ] || { echo "${LEFT}"; die "個人情報（ユーザー名・氏名・メールアドレス）が残っている（出力していない）"; }
done
BIG=$(find "${S}" -type f -size +5M)
[ -z "${BIG}" ] || { echo "${BIG}"; die "5 MB を超えるファイルがある（コード以外が混ざっていないか確かめる）"; }
echo "（gitleaks で秘密情報を検査中…）"
gitleaks dir "${S}" --no-banner --redact --exit-code 1 || die "gitleaks が秘密情報らしきものを検出した（出力していない）" 3

# ---- 出力（.git は残して中身だけ入れ替える）
mkdir -p "${OUT}" || die "出力先を作れない: ${OUT}"
rsync -a --delete --exclude '.git' "${S}/" "${OUT}/" || die "出力先へ写せない: ${OUT}"
N=$(find "${OUT}" -type f -not -path '*/.git/*' | wc -l | tr -d ' ')
echo "公開用フォルダ: ${OUT}（${N} ファイル・置き換えたファイル $(printf '%s' "${HITS}" | grep -c . || true) 本）"
( cd "${OUT}" && find . -maxdepth 1 -mindepth 1 -not -name '.git' | sort | sed 's|^\./|  |' )
