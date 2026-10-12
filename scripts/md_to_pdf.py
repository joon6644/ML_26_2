"""보고서 마크다운 → PDF (1차 보고서와 같은 모양: A4, 맑은 고딕, 헤드리스 Edge/Chrome으로 인쇄)

    .venv\\Scripts\\python.exe scripts/md_to_pdf.py docs/report/2차_예측모델_보고서.md
"""
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Malgun Gothic', sans-serif; font-size: 10pt; line-height: 1.7; color: #222; }
h1 { font-size: 17pt; border-bottom: 2px solid #222; padding-bottom: 6px; margin: 26px 0 14px; }
h1:first-of-type { font-size: 19pt; margin-top: 0; }
h2 { font-size: 13pt; border-left: 5px solid #2f6db5; padding-left: 10px; margin: 22px 0 10px; }
h3, h4 { font-size: 11pt; margin: 16px 0 8px; }
p { margin: 6px 0 10px; }
ul, ol { margin: 4px 0 10px; padding-left: 22px; }
li { margin: 3px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 5px 7px; vertical-align: top; }
th { background: #f0f3f7; font-weight: bold; }
pre { background: #f5f5f5; border: 1px solid #e2e2e2; padding: 9px 11px; font-family: Consolas, 'Malgun Gothic', monospace; font-size: 8.5pt; line-height: 1.5; white-space: pre-wrap; page-break-inside: avoid; }
code { font-family: Consolas, 'Malgun Gothic', monospace; font-size: 9pt; }
p code, li code, td code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
hr { border: none; border-top: 1px solid #ddd; margin: 22px 0; }
img { max-width: 100%; }
h1, h2, h3, h4 { break-after: avoid; }                          /* 제목이 페이지 끝에 혼자 남지 않게 */
p:has(+ table), p:has(+ pre), p:has(+ ul), p:has(+ ol) { break-after: avoid; }   /* 표·목록 앞 설명 줄도 함께 */
h1 + p, h2 + p { break-before: avoid; }
"""


def browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    return shutil.which("chrome") or shutil.which("msedge")


def main(md_path):
    md = Path(md_path).resolve()
    body = markdown.markdown(md.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    title = md.stem.replace("_", " ")
    html = md.with_suffix(".html")
    html.write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    pdf = md.with_suffix(".pdf")
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html.unlink()
    print(f"저장: {pdf}")


if __name__ == "__main__":
    main(sys.argv[1])
