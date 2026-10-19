"""講座「Physical AI 応用1」最終課題の動画（1〜3 分・等倍）を組み立てる。

素材:
  学習前   raw/parent_s42/<本>__FAIL.mp4（配布 kit の録画。各 task の最初の回）
  学習後   raw/m1_s43/<本>__SUCCESS.mp4（無ければ raw/m1_s42）・同じ場所の <ケチャップ>__SUCCESS.mp4
  人手デモ datasets/mdemo1/staging_manual20/*.npz（保存画像は上下逆なので、配布 kit の録画と同じ向きに反転する）
--placeholder-if-missing: 録画が無い素材を代わりの映像で埋め、全フレームに【仮】を入れる（出力名も _PLACEHOLDER）。
出力: final/course_final_video.mp4・SEGMENTS.json・確認用の CONTACT_SHEET.png
実行: PARC2026_pre-main/venv/bin/python work/tools/course_final/make_course_video.py [--placeholder-if-missing]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
CF = REPO / "work/reports/course_final_20261009"
RAW = CF / "raw"
MANUAL = REPO / "datasets/mdemo1/staging_manual20"
BOOK = "take_the_yellow_book_off_the_shelf_and_put_it_on_the_table"
KETCHUP = "LIVING_ROOM_SCENE1_take_the_ketchup_out_of_the_basket_and_put_it_on_the_table"
STANDIN_BEFORE = REPO / "work/reports/videos/t3_pathb12k" / f"{BOOK}__FAIL.mp4"
MANUAL_BOOK = "015__yellow_book__init029.npz"
MANUAL_KETCHUP = "004__V08_ketchup__init004.npz"
INSTRUCTION = "take the yellow book off the top of the shelf and put it back on the table on the left"

W, H, FPS = 1280, 720, 20
HOLD = FPS  # 各映像の最後の状態を 1 秒止める
BOX_X, BOX_Y, BOX_W, BOX_H = 128, 122, 1024, 512
BG, FG, SUB, ACCENT, RED, GRID = (24, 28, 34), (240, 242, 245), (190, 196, 204), (255, 196, 61), (235, 64, 52), (70, 76, 86)
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_R = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
_fonts: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}


def font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    if (bold, size) not in _fonts:
        _fonts[(bold, size)] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _fonts[(bold, size)]


def text_w(d: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont) -> int:
    left, _, right, _ = d.textbbox((0, 0), s, font=f)
    return right - left


def fit_font(d: ImageDraw.ImageDraw, s: str, bold: bool, size: int, max_w: int) -> ImageFont.FreeTypeFont:
    while size > 14 and text_w(d, s, font(bold, size)) > max_w:
        size -= 1
    return font(bold, size)


def centered(d: ImageDraw.ImageDraw, s: str, y: int, bold: bool, size: int, fill: tuple[int, int, int], cx: int = W // 2,
             max_w: int = W - 80) -> None:
    f = fit_font(d, s, bold, size, max_w)
    d.text((cx - text_w(d, s, f) // 2, y), s, font=f, fill=fill)


def stamp(d: ImageDraw.ImageDraw) -> None:
    s = "【仮】本番の録画ではありません"
    f = font(True, 20)
    d.text((W - 40 - text_w(d, s, f), 4), s, font=f, fill=RED)


def ffbin(name: str) -> str:
    p = shutil.which(name) or str(Path.home() / "miniforge3/bin" / name)
    if not Path(p).exists():
        sys.exit(f"★ {name} が見つからない")
    return p


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_mp4(path: Path) -> np.ndarray:
    probe = subprocess.run([ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    w, h = (int(x) for x in probe.stdout.split()[0].split(",")[:2])
    raw = subprocess.run([ffbin("ffmpeg"), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def read_manual(name: str) -> np.ndarray:
    z = np.load(MANUAL / name)
    return np.concatenate([z["front"][:, ::-1], z["wrist"][:, ::-1]], axis=2)


def clip_canvas(fr: np.ndarray | None, caption: str, sub: str, step: int | None, placeholder: bool) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 26), caption, font=fit_font(d, caption, True, 34, W - 120), fill=FG)
    if sub:
        d.text((60, 76), sub, font=fit_font(d, sub, False, 24, W - 120), fill=SUB)
    if fr is not None:
        h, w = fr.shape[:2]
        scale = min(BOX_W / w, BOX_H / h)
        nw, nh = int(w * scale), int(h * scale)
        v = Image.fromarray(np.ascontiguousarray(fr)).resize((nw, nh), Image.BICUBIC)
        img.paste(v, (BOX_X + (BOX_W - nw) // 2, BOX_Y + (BOX_H - nh) // 2))
        if w == 2 * h:  # 俯瞰と手先の 2 画面
            for label, cx in (("俯瞰カメラ", BOX_X + BOX_W // 4), ("手先カメラ", BOX_X + 3 * BOX_W // 4)):
                centered(d, label, 644, False, 22, SUB, cx=cx, max_w=BOX_W // 2)
    if step is not None:
        d.text((60, 680), f"ステップ {step} / 上限 300", font=font(False, 22), fill=SUB)
    foot = "シミュレーション(LIBERO)・等倍再生"
    d.text((W - 60 - text_w(d, foot, font(False, 20)), 684), foot, font=font(False, 20), fill=SUB)
    if placeholder:
        stamp(d)
    return np.asarray(img)


def title_card(placeholder: bool) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, "人手デモで「元に戻す」作業をロボットに教える", 230, True, 46, FG)
    centered(d, "Physical AI 応用1 最終課題", 312, False, 30, ACCENT)
    centered(d, "題材: PARC2026 本選 Track3(逆転タスク) / 環境: LIBERO / モデル: π0.5", 400, False, 24, SUB)
    centered(d, "※動画はすべて等倍再生です(シミュレーションの1ステップ = 0.05秒)", 450, False, 22, SUB)
    if placeholder:
        stamp(d)
    return np.asarray(img)


def results_card(placeholder: bool) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    centered(d, "結果(同じ評価条件で各20回)", 60, True, 38, FG)
    cols = [600, 840, 1090]
    heads = [["学習前"], ["機械データだけで", "追加学習"], ["人手デモを混ぜて", "追加学習"]]
    for cx, lines in zip(cols, heads):
        y0 = 166 if len(lines) == 1 else 150
        for i, s in enumerate(lines):
            centered(d, s, y0 + i * 32, False, 24, SUB, cx=cx, max_w=240)
    rows = [("本のタスク", ["0/20", "0/20", "13/20"], True),
            ("ケチャップのタスク", ["20/20", "20/20", "16/20"], False),
            ("衝突(計40回中)", ["15回", "19回", "0回"], True)]
    y = 250
    for label, vals, hl in rows:
        d.line([(80, y - 22), (W - 80, y - 22)], fill=GRID, width=2)
        d.text((100, y), label, font=font(False, 30), fill=FG)
        for i, (cx, v) in enumerate(zip(cols, vals)):
            strong = hl and i == 2
            centered(d, v, y - 2, strong, 36 if strong else 32, ACCENT if strong else FG, cx=cx, max_w=240)
        y += 86
    d.line([(80, y - 22), (W - 80, y - 22)], fill=GRID, width=2)
    centered(d, "人手デモを混ぜたことで、本のタスクに成功するようになり、衝突もなくなった。", y + 30, False, 26, FG)
    centered(d, "一方で動きは遅くなった(本の成功に240〜286ステップ。上限は300)。", y + 76, False, 26, SUB)
    if placeholder:
        stamp(d)
    return np.asarray(img)


def pick_clip(dirs: list[str], name: str, tag: str) -> Path | None:
    for dn in dirs:
        p = RAW / dn / f"{name}__{tag}.mp4"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placeholder-if-missing", action="store_true")
    a = ap.parse_args()
    for f in (FONT_B, FONT_R):
        if not Path(f).exists():
            sys.exit(f"★ 日本語フォントが無い: {f}")

    before = pick_clip(["parent_s42"], BOOK, "FAIL")
    after_book = pick_clip(["m1_s43", "m1_s42"], BOOK, "SUCCESS")
    after_ketchup = pick_clip(["m1_s43", "m1_s42"], KETCHUP, "SUCCESS")
    missing = [n for n, p in (("学習前", before), ("学習後の本", after_book), ("学習後のケチャップ", after_ketchup)) if p is None]
    placeholder = bool(missing)
    if missing and not a.placeholder_if_missing:
        sys.exit(f"★ 録画が無い: {', '.join(missing)}（先に record_policy_clips.sh を回す。試しに組むなら --placeholder-if-missing）")

    manual_book, manual_ketchup = read_manual(MANUAL_BOOK), read_manual(MANUAL_KETCHUP)
    src_before = before or STANDIN_BEFORE
    fr_before = read_mp4(src_before)
    fr_after_book = read_mp4(after_book) if after_book else manual_book
    fr_after_ketchup = read_mp4(after_ketchup) if after_ketchup else manual_ketchup

    def clip(frames: np.ndarray, caption: str, sub: str) -> Iterator[np.ndarray]:
        last = None
        for i, fr in enumerate(frames):
            last = clip_canvas(fr, caption, sub, i, placeholder)
            yield last
        for _ in range(HOLD):
            yield last

    def static(img: np.ndarray, seconds: float) -> Iterator[np.ndarray]:
        for _ in range(int(seconds * FPS)):
            yield img

    segs = [
        ("タイトル", "カード", static(title_card(placeholder), 6)),
        ("タスクの説明", f"人手デモ {MANUAL_BOOK} の最初の画面",
         static(clip_canvas(manual_book[0], "タスク: 棚の上の本を取り、机の上の決められた場所に置く", f'指示文: "{INSTRUCTION}"', None,
                            placeholder), 6)),
        ("学習前", str(src_before.relative_to(REPO)),
         clip(fr_before, "① 学習前: 機械的な教師データだけで学習したモデル", "本を決められた場所に置けず、上限300ステップで失敗(本のタスクは20回中0回成功)")),
        ("人手デモ(本)", f"datasets/mdemo1/staging_manual20/{MANUAL_BOOK}",
         clip(manual_book, "② 人手デモ: キーボードで操作して記録", "先に腕を上げてから本を運ぶ(本10本・ケチャップ10本の計20本を約35分で収録)")),
        ("人手デモ(ケチャップ)", f"datasets/mdemo1/staging_manual20/{MANUAL_KETCHUP}",
         clip(manual_ketchup, "② 人手デモ: 自作の逆転タスク", "「かごからケチャップを取り出して机に置く」(既存のタスクを逆向きにして自動生成)")),
        ("学習後(本)", str(after_book.relative_to(REPO)) if after_book else f"【仮】{MANUAL_BOOK}",
         clip(fr_after_book, "③ 学習後: 人手デモを25%混ぜて追加学習", "自分で本を運んで成功(本のタスクは20回中13回成功)")),
        ("学習後(ケチャップ)", str(after_ketchup.relative_to(REPO)) if after_ketchup else f"【仮】{MANUAL_KETCHUP}",
         clip(fr_after_ketchup, "③ 学習後: 逆転タスクも自分で実行", "ケチャップのタスクは20回中16回成功(学習前の20回中20回からは低下)")),
        ("結果", "カード", static(results_card(placeholder), 10)),
    ]

    out_dir = CF / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "course_final_video" + ("_PLACEHOLDER" if placeholder else "")
    out, tmp = out_dir / f"{stem}.mp4", out_dir / f"{stem}.tmp.mp4"
    cmd = [ffbin("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    record, thumbs, total = [], [], 0
    for name, source, frames in segs:
        n, samples = 0, []
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
            if n % 20 == 0:
                samples.append(np.asarray(Image.fromarray(fr).resize((320, 180), Image.BICUBIC)))
            n += 1
        thumbs.append((name, samples[len(samples) // 2]))
        total += n
        record.append({"segment": name, "seconds": round(n / FPS, 2), "source": source})
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("★ ffmpeg が失敗した")
    seconds = total / FPS
    if not 60 <= seconds <= 180:
        sys.exit(f"★ 長さが {seconds:.1f} 秒（1〜3 分に収まらない）。{tmp} は残す")
    tmp.replace(out)

    sheet = Image.new("RGB", (4 * 320, 2 * 200), BG)
    d = ImageDraw.Draw(sheet)
    for i, (name, fr) in enumerate(thumbs):
        x, y = (i % 4) * 320, (i // 4) * 200
        sheet.paste(Image.fromarray(fr), (x, y))
        d.text((x + 6, y + 180), name, font=font(False, 16), fill=FG)
    sheet.save(out_dir / f"{stem}_CONTACT_SHEET.png")
    sources = {}
    for p in [src_before, after_book, after_ketchup, MANUAL / MANUAL_BOOK, MANUAL / MANUAL_KETCHUP]:
        if p is not None:
            sources[str(Path(p).relative_to(REPO))] = sha256(Path(p))
    meta = {"video": str(out.relative_to(REPO)), "seconds": round(seconds, 2), "fps": FPS, "size": f"{W}x{H}",
            "placeholder": placeholder, "missing": missing, "segments": record, "source_sha256": sources,
            "video_sha256": sha256(out)}
    (out_dir / f"{stem}_SEGMENTS.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if not placeholder:  # 本番の動画ができたら【仮】版は紛らわしいので片付ける
        for p in out_dir.glob("course_final_video_PLACEHOLDER*"):
            p.unlink()
            print(f"【仮】版を削除: {p.name}")
    print(f"動画: {out}  {seconds:.1f} 秒" + ("  ★【仮】版（録画がまだ無い）" if placeholder else ""))
    for r in record:
        print(f"  {r['segment']:<12} {r['seconds']:>6.2f} 秒  {r['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
