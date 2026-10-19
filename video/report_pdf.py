"""講座「Physical AI 応用1」最終課題のレポート（Markdown）を A4 の PDF にする。

pandoc（Markdown → HTML）→ Chrome headless（HTML → PDF）。競技レポート用の work/tools/report_to_pdf.py は
2 ページに詰める CSS で、pandoc 既定テンプレートの印刷用 CSS が本文だけを 12pt に戻すため表が極端に小さくなる。
ここでは本文と表を同じ大きさにそろえる（!important で既定テンプレートを上書き）。ページ数の上限は設けない。
使い方: python3 work/tools/course_final/report_pdf.py <input.md> <output.pdf>
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 16mm 17mm; }
html, body { font-size: 10.5pt !important; }
body { font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif; max-width: none !important;
       margin: 0 !important; padding: 0 !important; line-height: 1.65; color: #111; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; padding-bottom: 4pt; border-bottom: 1pt solid #333; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 10.5pt !important; margin: 8pt 0; }
th, td { border: 0.6pt solid #999; padding: 4pt 6pt; text-align: center; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef2f8; font-weight: 600; }
"""


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if not Path(CHROME).exists():
        sys.exit(f"★ Chrome が無い: {CHROME}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        header, html = Path(td) / "header.html", Path(td) / "report.html"
        header.write_text(f"<style>{CSS}</style>\n", encoding="utf-8")
        subprocess.run(["pandoc", "--from=gfm", "--to=html5", "--standalone", f"--include-in-header={header}",
                        "--metadata=pagetitle:report", "-o", str(html), str(src)], check=True)
        tmp = out.with_suffix(".tmp.pdf")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={tmp}", html.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not tmp.exists() or tmp.stat().st_size == 0:
        sys.exit("★ PDF が作られなかった")
    tmp.replace(out)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    print(f"レポート PDF: {out}（{pages} ページ）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
