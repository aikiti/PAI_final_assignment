#!/usr/bin/env bash
# =============================================================================
# 講座「Physical AI 応用1」最終課題の提出物を 1 コマンドで作る。
#   bash work/tools/course_final/run_all.sh          … 全部（録画〔約 25 分・Mac が重い〕→ 動画 → 公開用のコードのフォルダ → レポート PDF）
#   bash work/tools/course_final/run_all.sh --light  … 録画以外（モデルを動かさない。録画が無ければ動画は【仮】版）
# 録画の条件（AC 電源・本選の評価や学習が動いていない等）は record_policy_clips.sh が起動前に確かめ、違えば止まる。
# 出力: work/reports/course_final_20261009/final/ に 動画（YouTube へ）・レポート PDF（omnicampus へ）。
#   コードは ~/PhysicalAI/PAI_final_assignment/（GitHub へ。公開は本人の GO の後）。手順書は final/ と同じ階層の README.md。
# =============================================================================
set -u -o pipefail
REPO=/Users/USERNAME/PhysicalAI/PARC2026
cd "${REPO}" || exit 1
CF="${REPO}/work/reports/course_final_20261009"
VPY=PARC2026_pre-main/venv/bin/python
MODE="${1:-all}"
case "${MODE}" in all|--light) ;; *) sed -n '2,8p' "$0"; exit 2 ;; esac
die() { echo "★ $1"; exit "${2:-1}"; }
mkdir -p "${CF}/logs" "${CF}/final"
exec > >(tee -a "${CF}/logs/run_all_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "=== 講座課題 ${MODE} $(date '+%F %T') ==="
if [ "${MODE}" = "all" ]; then
  bash work/tools/course_final/record_policy_clips.sh || die "録画で止まった（上の ★ の理由を見る）"
  "${VPY}" work/tools/course_final/make_course_video.py || die "動画の組み立てに失敗"
else
  "${VPY}" work/tools/course_final/make_course_video.py --placeholder-if-missing || die "動画の組み立てに失敗"
fi
bash work/tools/course_final/make_public_repo.sh || die "公開用のコードのフォルダの作成に失敗"
python3 work/tools/course_final/report_pdf.py "${CF}/REPORT_COURSE_FINAL.md" "${CF}/final/説明資料_PAI最終課題_Shinji_0001.pdf" \
  || die "レポート PDF の作成に失敗"
echo "=== 講座課題 DONE $(date '+%F %T') ==="
ls -la "${CF}/final"
