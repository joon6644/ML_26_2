# %% [markdown]
# # 피처 기술통계량 (가격 제외)
# 파일·계열별 수치 컬럼의 개수, 결측률, 평균, 표준편차, 분위수, 왜도.
# 실행: `python eda/02_descriptive_stats.py` → eda/tables/*.csv, eda/02_descriptive_stats.md

# %%
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
TAB = ROOT / "eda" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
START = "2016-01-01"


def rd(path):
    return pd.read_parquet(P / path)


def describe(df, cols, by=None):
    """수치 컬럼 요약. by가 있으면 그룹별로."""
    def one(s):
        s = pd.to_numeric(s, errors="coerce")
        v = s.dropna()
        return pd.Series({
            "n": len(s), "결측%": s.isna().mean() * 100, "평균": v.mean(), "표준편차": v.std(), "최소": v.min(),
            "25%": v.quantile(.25), "중앙값": v.median(), "75%": v.quantile(.75), "최대": v.max(), "왜도": v.skew(),
        })
    rows = []
    groups = df.groupby(by, observed=True) if by else [(None, df)]
    for key, g in groups:
        for c in cols:
            r = one(g[c])
            r["변수"] = c if key is None else f"{key if not isinstance(key, tuple) else ' / '.join(map(str, key))} · {c}"
            rows.append(r)
    out = pd.DataFrame(rows).set_index("변수")
    return out


SECTIONS = []  # (제목, 설명, 표)


def add(title, note, table, name):
    table.to_csv(TAB / f"{name}.csv", encoding="utf-8-sig")
    SECTIONS.append((title, note, table))
    print(f"[{name}] {len(table)}행")


# %% [markdown]
# ## 공통

# %%
w = rd("common/weather_national_daily.parquet")
add("기상 (전국 평균, 일)", "ASOS 관측소 평균. rain은 무강수=0, snow_depth·rain_1h_max는 관측이 있는 날만 값이 있음",
    describe(w, [c for c in w.columns if c != "date"]), "weather_national")

st = rd("common/weather_station_daily.parquet")
add("기상 (관측소별 일자료 전체 풀링)", "관측소 97곳 × 일. 지역 간 편차가 포함된 분포",
    describe(st, ["temp_avg", "temp_min", "temp_max", "rain", "sunshine_hr", "humidity", "wind_avg", "wind_gust"]), "weather_station")

fx = rd("common/fx_daily.parquet")
fx["연도"] = fx.date.dt.year
add("환율 (원/달러, 일)", "연도별", describe(fx, ["usd_krw"], by="연도"), "fx_by_year")

m = rd("common/macro_monthly.parquet")
sel = m[(m.table.isin(["intl_commodity", "fuel_price"])) | ((m.table == "cpi") & m.name.isin(["총지수", "식료품", "배추", "무", "양파", "파", "돼지고기", "국산쇠고기", "달걀", "고등어"]))]
add("거시지표 (월)", "국제상품가격(달러 단위), 주유소 가격(원/ℓ), 소비자물가지수(2020=100)",
    describe(sel, ["value"], by=["table", "name"]), "macro")

# %% [markdown]
# ## 농산

# %%
tr = rd("agri/trade_by_item_daily.parquet")
top = tr.groupby("item_nm").qty_kg.sum().nlargest(15).index
t = tr[tr.item_nm.isin(top)].assign(qty_ton=lambda d: d.qty_kg / 1000)
add("가락시장 거래 (물량 상위 15개 품목, 일)", "qty_ton = kg 단위 거래 물량(톤), avg_price_kg = 원/kg. 거래가 있는 날만 (2018~)",
    describe(t, ["qty_ton", "avg_price_kg"], by="item_nm"), "trade_by_item")

imp = rd("agri/import_monthly.parquet")
imp["hs2"] = imp.hs_code.str[:2]
mon = imp.groupby(["hs2", "month"]).agg(imp_ton=("imp_kg", lambda x: x.sum() / 1000), imp_musd=("imp_usd", lambda x: x.sum() / 1e6)).reset_index()
add("농산 수입 (HS 2단위 월 합계)", "imp_ton = 톤, imp_musd = 백만 달러. 07 채소, 08 과일, 09 향신료, 10 곡물, 12 유지종자",
    describe(mon, ["imp_ton", "imp_musd"], by="hs2"), "agri_import")

pr = rd("agri/production_annual.parquet")
keys = ["배추:생산량", "무:생산량", "양파:생산량", "마늘:생산량", "건고추:생산량", "사과:생산량", "배:생산량", "감귤:생산량", "미곡:생산량"]
pr = pr[pr.item_nm.isin(keys) & (pr["시도별"].isin(["전국", "계"]))]
add("농작물 생산량 (전국, 연)", "톤. 2016~2025", describe(pr, ["value"], by="item_nm"), "agri_production")

# %% [markdown]
# ## 축산

# %%
a = rd("livestock/auction_monthly.parquet")
a = a[(a.market_cd == "CTot") & (a.grade == "평균")]
add("소·돼지 경락 (전국, 평균 등급, 월)", "price = 원/kg(도체), head_count = 경락 두수(월 합계)",
    describe(a, ["price", "head_count"], by="species"), "auction")

p = rd("livestock/pig_rep_price_daily.parquet")
add("돈육 대표가격 (일)", "원/kg. 박피는 거래가 적은 날의 이상 저가 포함", describe(p, ["price"], by="skin"), "pig_rep_price")

s = rd("livestock/stock_monthly.parquet")
add("축산물 재고 (월)", "ton (소는 kg → ton으로 변환). 2019-05~", describe(s, ["totStock"], by="species"), "stock")

ce = rd("livestock/census_quarterly.parquet")
ce = ce[(ce["시도별"] == "전국") & (ce["사육규모별"].fillna("합계") == "합계") & ce.item_nm.str.contains("마리수")]
add("사육두수 (전국, 분기)", "마리", describe(ce, ["value"], by=["table", "item_nm"]), "census")

d = rd("livestock/disease_events.parquet")
d = d[d.date >= START]
d["연도"] = d.date.dt.year
ev = d[d.disease_std.isin(["고병원성조류인플루엔자", "아프리카돼지열병", "구제역", "럼피스킨병"])]
add("주요 가축질병 발생 (건별, 2016~)", "head_count = 발생 두수(건당)", describe(ev, ["head_count"], by="disease_std"), "disease")

# %% [markdown]
# ## 수산

# %%
ct = rd("fishery/sea_temp_daily.parquet")
add("해역별 수온 (일)", "℃. 기상청 해양기상부이 중 2016년부터 운영된 17곳의 일평균을 해역별로 평균",
    describe(ct, ["water_temp"], by="sea"), "sea_temp")

fp = rd("fishery/production_monthly.parquet")
fp = fp[(fp["어업별"] == "계") & fp["품종별"].isin(["고등어", "살오징어(오징어)", "멸치", "갈치", "참조기", "꽃게", "김류", "굴류", "전복류", "바지락"])]
fp = fp.assign(지표=fp.item_id.map({"T01": "생산량(톤)", "T05": "생산금액(천원)"}))
add("어업생산 (주요 품종, 월)", "2016-01 ~ 2026-08", describe(fp, ["value"], by=["품종별", "지표"]), "fishery_production")

r = rd("fishery/redtide_events.parquet")
add("적조 발생 (보고 건별)", "밀도 = 개체/mL", describe(r, ["density_max", "water_temp_max"]), "redtide")

# %% [markdown]
# ## 마크다운 저장

# %%
def fmt(x):
    if pd.isna(x):
        return ""
    a = abs(x)
    return f"{x:,.0f}" if a >= 1000 else f"{x:,.1f}" if a >= 10 else f"{x:,.2f}"


lines = ["# 피처 기술통계량 (가격 제외)", "",
         "> 자동 생성: `python eda/02_descriptive_stats.py` · 원본 CSV: `eda/tables/` · 해석: [01_features_eda.md](01_features_eda.md)", ""]
for title, note, tab in SECTIONS:
    t = tab.copy()
    t["n"] = t["n"].map(lambda v: f"{v:,.0f}")
    t["결측%"] = t["결측%"].map(lambda v: f"{v:.1f}")
    for c in ["평균", "표준편차", "최소", "25%", "중앙값", "75%", "최대", "왜도"]:
        t[c] = t[c].map(fmt)
    lines += [f"## {title}", "", f"_{note}_", "", t.reset_index().to_markdown(index=False), ""]
(ROOT / "eda" / "02_descriptive_stats.md").write_text("\n".join(lines), encoding="utf-8")
print("저장: eda/02_descriptive_stats.md")
