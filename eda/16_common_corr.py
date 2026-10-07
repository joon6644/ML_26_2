# %% [markdown]
# # 공통 피처 상관관계 (학습 기간 2016~2022, Spearman)
# 1) 공통 피처끼리: 분야별 상관행렬 → |ρ| ≥ 0.8 쌍, 농산물 히트맵
# 2) 공통 피처 ↔ 타깃 y: 분야별 Spearman
# 날짜 단위 변수(기상·경제)는 모든 행에 같은 값이 반복되므로, 피처끼리 상관은 행 가중이 아닌 '날짜 1행'으로도 따로 계산
# 실행: `python eda/16_common_corr.py` → eda/16_common_corr.md, docs/report/figures/16_common_corr.png

# %%
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_features as bf  # noqa: E402

P = ROOT / "data" / "processed"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
COMMON = bf.COMMON + bf.HIST + ["woy"]                       # 수치형 공통 피처 (범주형 3개 제외)
DATE_LEVEL = ["dow", "woy"] + bf.WX + [c for c in bf.ECON if c != "cpi_item"]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 150, "font.size": 7})


def pairs(c, thr=0.8):
    cols = c.columns
    out = [(a, b, c.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(c.loc[a, b]) >= thr]
    return sorted(out, key=lambda x: -abs(x[2]))


def main():
    L = ["# 공통 피처 상관관계 (학습 기간 2016~2022, Spearman)", "", "> 자동 생성: `python eda/16_common_corr.py`", ""]
    ycorr, frames = {}, {}
    for d, nm in NAMES.items():
        f = pd.read_parquet(P / d / "features.parquet")
        f = f[(f.date >= "2016-01-01") & (f.date <= "2022-12-31")]
        frames[d] = f
        ycorr[nm] = f[COMMON].corrwith(f.y, method="spearman")
    # 날짜 단위 변수: 날짜 1행 기준 (분야와 무관하게 같은 값)
    dd = frames["agri"].groupby("date")[DATE_LEVEL].first()
    cd = dd.corr(method="spearman")
    L += [f"## 1. 날짜 단위 변수끼리 (날짜 {len(dd):,}개 기준, |ρ| ≥ 0.8)", "", "| 변수 A | 변수 B | ρ |", "|---|---|---|"]
    L += [f"| {a} | {b} | {v:+.2f} |" for a, b, v in pairs(cd)] + [""]
    # 시계열 단위 변수 포함 전체: 분야별 행 기준
    for d, nm in NAMES.items():
        c = frames[d][COMMON].corr(method="spearman")
        pr = [(a, b, v) for a, b, v in pairs(c) if not (a in DATE_LEVEL and b in DATE_LEVEL)]
        L += [f"## 2. {nm}: 가격·가격이력·교역 변수가 포함된 쌍 (|ρ| ≥ 0.8)", "", "| 변수 A | 변수 B | ρ |", "|---|---|---|"]
        L += [f"| {a} | {b} | {v:+.2f} |" for a, b, v in pr] or ["| (없음) | | |"]
        L.append("")
        if d == "agri":
            heat = c
    y = pd.DataFrame(ycorr).round(3)
    y["|ρ| 최대"] = y.abs().max(axis=1)
    y = y.sort_values("|ρ| 최대", ascending=False)
    L += ["## 3. 공통 피처 ↔ 타깃 y (분야별 Spearman ρ)", "", y.reset_index().rename(columns={"index": "피처"}).to_markdown(index=False), ""]
    (ROOT / "eda" / "16_common_corr.md").write_text("\n".join(L), encoding="utf-8")
    y.to_csv(ROOT / "eda" / "tables" / "common_corr_y.csv", encoding="utf-8-sig")

    order = bf.HIST + ["price_kg", "import_kg", "import_usd", "export_kg", "cpi_item"] + [c for c in DATE_LEVEL if c != "cpi_item"]
    order = [c for c in order if c in heat]
    h = heat.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(h.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_yticks(range(len(order)), order)
    for k in (len(bf.HIST), len(bf.HIST) + 5):
        ax.axhline(k - 0.5, color="#555", lw=0.6)
        ax.axvline(k - 0.5, color="#555", lw=0.6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("공통 피처 상관 (농산물, 학습 2016~2022)  |  가격이력 · 가격·교역 · 날짜 단위(달력·기상·경제)")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "report" / "figures" / "16_common_corr.png")
    print("\n".join(L))


if __name__ == "__main__":
    main()
