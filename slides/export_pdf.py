"""Export a deck's source (deck.json + slides/*.html) to PDF with headless Google Chrome.

    python3 slides/export_pdf.py slides/02_results_choysum/source slides/02_results_choysum/Results_Choysum.pdf

Speaker notes (<aside>) are left out. Pass --png DIR to also save one PNG per slide.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """@page{size:1920px 1080px;margin:0}*{margin:0;padding:0;box-sizing:border-box}
body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
section{position:relative;width:1920px;height:1080px;overflow:hidden;break-after:page}
section:last-child{break-after:auto}
ul{padding-left:1.2em} li{margin-bottom:8px} table{border-collapse:collapse}
td,th{padding:0.35em 0.6em;border-bottom:1px solid #D9D7D5;vertical-align:top}
x-shape{display:block;flex:none}
x-shape[kind=arrow-right]{clip-path:polygon(0 30%,60% 30%,60% 0,100% 50%,60% 100%,60% 70%,0 70%)}"""


def align_columns(html: str) -> str:
    """Give each <td> its header's text-align, as the deck viewer does."""
    def fix(table):
        rows = re.findall(r"<tr.*?</tr>", table, flags=re.S)
        aligns = [re.search(r"text-align:(\w+)", th).group(1) if "text-align" in th else "left"
                  for th in re.findall(r"<th[^>]*>", rows[0])]
        for r in rows[1:]:
            cells = iter(aligns)
            table = table.replace(r, re.sub(r"<td>", lambda m: f'<td style="text-align:{next(cells, "left")}">', r), 1)
        return table
    return re.sub(r"<table.*?</table>", lambda m: fix(m.group(0)), html, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--png", type=Path)
    args = ap.parse_args()

    deck = json.loads((args.source / "deck.json").read_text())
    slides = [align_columns(re.sub(r"<aside>.*?</aside>", "", (args.source / "slides" / f"{sid}.html").read_text(), flags=re.S))
              for sid in deck["order"]]
    fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    head = f'<!doctype html><html><head><meta charset="utf-8"><title>{deck["title"]}</title>{fonts}<style>{CSS}</style></head><body>'
    chrome = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=10000"]

    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "deck.html"
        page.write_text(head + "".join(slides) + "</body></html>")
        subprocess.run(chrome + ["--no-pdf-header-footer", f"--print-to-pdf={args.pdf.resolve()}", page.as_uri()],
                       check=True, capture_output=True)
        if args.png:
            args.png.mkdir(parents=True, exist_ok=True)
            for i, s in enumerate(slides, 1):
                one = Path(tmp) / f"slide_{i:02d}.html"
                one.write_text(head + s + "</body></html>")
                subprocess.run(chrome + ["--window-size=1920,1080", f"--screenshot={(args.png / one.with_suffix('.png').name).resolve()}",
                                         one.as_uri()], check=True, capture_output=True)
    print(f"{len(slides)} slides -> {args.pdf}")


if __name__ == "__main__":
    main()
