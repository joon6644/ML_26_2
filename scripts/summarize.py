"""data/processed/ 현황을 docs/processed_status.md 로 정리한다 (build.py 다음에 실행).

    python scripts/summarize.py
"""
import datetime as dt
import glob
from pathlib import Path

import pandas as pd

from common import PROCESSED, ROOT

TIME_COLS = ("date", "month", "period")
DOMAINS = [("common", "공통"), ("agri", "농산물"), ("livestock", "축산물"), ("fishery", "수산물")]


def describe(path):
    df = pd.read_parquet(path)
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    start = end = "-"
    if tcol:
        t = df[tcol].dropna()
        if len(t):
            start, end = (str(t.min())[:10], str(t.max())[:10]) if tcol != "period" else (t.min(), t.max())
    cand = [c for c in ("item_nm", "hs_name", "stn_nm", "disease", "name", "gds_mclsf_nm") if c in df.columns]
    cand += [c for c in df.columns if c.endswith("별")]  # KOSIS 분류 차원 (품종별, 시도별 …)
    key = max(cand, key=lambda c: df[c].nunique()) if cand else None  # 가장 세분화된 구분
    return {
        "파일": Path(path).stem, "행": f"{len(df):,}", "열": df.shape[1], "시작": start, "끝": end,
        "구분 수": f"{key} {df[key].nunique():,}" if key else "-",
        "크기(MB)": f"{Path(path).stat().st_size / 1e6:.1f}",
    }


def price_detail(domain):
    path = PROCESSED / domain / "price.parquet"
    if not path.exists():
        return ""
    df = pd.read_parquet(path, columns=["date", "item_nm"])
    g = df.groupby("item_nm").date.agg(["min", "max", "count"]).reset_index().sort_values("count", ascending=False)
    lines = ["| 품목 | 시작 | 끝 | 행 |", "|---|---|---|---|"]
    lines += [f"| {r.item_nm} | {r['min']:%Y-%m-%d} | {r['max']:%Y-%m-%d} | {r['count']:,} |" for _, r in g.iterrows()]
    return "\n".join(lines)


def main():
    out = [f"# 가공 데이터 현황\n\n> 자동 생성: `python scripts/summarize.py` ({dt.datetime.now():%Y-%m-%d %H:%M})\n"
           "> 컬럼 설명은 [processed_data.md](processed_data.md). 수집이 진행 중이면 행 수·기간이 늘어난다.\n"]
    for dom, name in DOMAINS:
        files = sorted(glob.glob(str(PROCESSED / dom / "*.parquet")))
        out.append(f"## {name} (`{dom}/`)\n")
        if not files:
            out.append("(아직 없음)\n")
            continue
        rows = [describe(f) for f in files]
        cols = list(rows[0])
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        detail = price_detail(dom) if dom != "common" else ""
        if detail:
            out.append(f"\n<details><summary>{name} 가격 품목별 기간</summary>\n\n{detail}\n\n</details>")
        out.append("")
    path = ROOT / "docs" / "processed_status.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
