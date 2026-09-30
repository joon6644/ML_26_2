# %% [markdown]
# # 피처 EDA (가격 제외)
# 가격(타깃)을 뺀 피처 파일들의 기간·결측·분포·계절성·가격과의 관계를 본다.
# 실행: `python eda/01_features_eda.py` (그림은 eda/figures/ 에 저장) 또는 VS Code에서 셀 단위 실행.
# 결과 해석은 eda/01_features_eda.md.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
FIG = ROOT / "eda" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 참고 팔레트(categorical 고정 순서) - dataviz 스킬 references/palette.md
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 150,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> figures/{name}.png")


def rd(path):
    return pd.read_parquet(P / path)


# %% [markdown]
# ## 0. 파일별 제공 기간 (커버리지)

# %%
cover = []
for f in sorted(P.glob("*/*.parquet")):
    if f.stem in ("price", "recipe_basic", "recipe_ingredient", "calendar"):
        continue
    d = pd.read_parquet(f)
    col = next((c for c in ("date", "month", "period") if c in d.columns), None)
    if col is None:
        continue
    t = d[col]
    if col == "period":
        t = pd.to_datetime(t.astype(str).str[:4] + "-" + t.astype(str).str[4:6].replace("", "01").where(t.astype(str).str.len() > 4, "01") + "-01", errors="coerce")
    t = t[(t >= "2016-01-01")]
    cover.append((f"{f.parent.name}/{f.stem}", t.min(), t.max(), len(d)))
cover = pd.DataFrame(cover, columns=["file", "start", "end", "rows"]).sort_values(["start", "file"], ascending=[False, True])
print(cover.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 0.32 * len(cover) + 1))
dom_color = {"common": SERIES[0], "agri": SERIES[2], "livestock": SERIES[1], "fishery": SERIES[6]}
for i, r in enumerate(cover.itertuples()):
    ax.barh(i, (r.end - r.start).days, left=r.start, height=0.55, color=dom_color[r.file.split("/")[0]])
ax.set_yticks(range(len(cover)), cover.file, fontsize=8)
ax.axvline(pd.Timestamp("2016-01-01"), color=INK2, lw=1, ls=":")
ax.set_title("피처 파일별 제공 기간 (2016-01 이후)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in dom_color.values()]
ax.legend(handles, ["공통", "농산", "축산", "수산"], loc="lower left", ncol=4, fontsize=8)
save(fig, "00_coverage")

# %% [markdown]
# ## 1. 공통: 기상 극값

# %%
st = rd("common/weather_station_daily.parquet")
st["year"] = st.date.dt.year
st = st.assign(heat=st.temp_max >= 33, cold=st.temp_min <= -12, heavy=st.rain >= 80)
yr = st.groupby(["year", "stn_id"])[["heat", "cold", "heavy"]].sum().groupby("year").mean()  # 관측소별 실제 일수의 평균
print(yr.round(1))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True)
for ax, (col, title), c in zip(axes, [("heat", "폭염일 (최고 ≥33℃)"), ("cold", "한파일 (최저 ≤ -12℃)"), ("heavy", "호우일 (일강수 ≥80mm)")], SERIES):
    ax.bar(yr.index, yr[col], color=c, width=0.7)
    ax.set_title(title)
    ax.set_ylabel("관측소 평균 일수/년")
fig.suptitle("연도별 기상 극값 (ASOS 97개 관측소 평균, 2026은 9월까지)", fontsize=11, color=INK2)
save(fig, "01_weather_extremes")

# %% [markdown]
# ## 2. 공통: 거시지표 (2016-01 = 100)

# %%
m = rd("common/macro_monthly.parquet")
ic = m[m.table == "intl_commodity"].pivot_table(index="month", columns="name", values="value")
fx = rd("common/fx_daily.parquet").set_index("date").usd_krw.resample("MS").mean()
idx = pd.concat([ic[["원유- Dubai", "옥수수", "소맥", "대두"]], fx.rename("원/달러 환율")], axis=1).dropna()
idx = idx / idx.iloc[0] * 100
fig, ax = plt.subplots(figsize=(10, 4))
for c, col in zip(SERIES, idx.columns):
    ax.plot(idx.index, idx[col], color=c, label=col)  # 5개 계열: 끝 라벨은 겹쳐서 범례로만 식별
ax.axhline(100, color=INK2, lw=1, ls=":")
ax.set_title("국제 원자재·환율 지수 (2016-01 = 100, 월)")
ax.legend(ncol=5, fontsize=8, loc="upper left")
save(fig, "02_macro_index")

cpi = m[m.table == "cpi"].pivot_table(index="month", columns="name", values="value")
vol = (cpi.pct_change(12, fill_method=None) * 100).std().drop(["총지수", "식료품", "식료품 및 비주류음료"], errors="ignore").sort_values().tail(15)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(vol.index, vol.values, color=SERIES[0], height=0.6)
for i, v in enumerate(vol.values):
    ax.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=8, color=INK2)
ax.set_title("소비자물가 품목별 전년동월비 변동성 (표준편차, %p)")
save(fig, "03_cpi_volatility")

# %% [markdown]
# ## 3. 농산: 가락시장 거래량

# %%
tr = rd("agri/trade_by_item_daily.parquet")
tr = tr[~tr.item_nm.isin(["알배기배추", "절임배추", "깐마늘(국산)", "키위"])]  # 같은 가락 품목을 공유하는 중복 매핑 제외
tr["month"] = tr.date.dt.month
items = ["배추", "무", "양파", "사과", "수박"]
fig, ax = plt.subplots(figsize=(9, 4))
for c, it in zip(SERIES, items):
    mo = tr[tr.item_nm == it].groupby("month").qty_kg.mean()
    ax.plot(mo.index, mo / mo.mean(), color=c, marker="o", ms=4, label=it)
ax.axhline(1, color=INK2, lw=1, ls=":")
ax.set_xticks(range(1, 13), [f"{i}월" for i in range(1, 13)])
ax.set_ylabel("월평균 물량 / 연평균")
ax.set_title("가락시장 일 거래량의 계절 패턴 (2018~2026)")
ax.legend(ncol=5, fontsize=8)
save(fig, "04_trade_seasonality")

rows = []
for it, x in tr.groupby("item_nm"):
    x = x.set_index("date").sort_index()
    x = x[(x.qty_kg > 0) & (x.avg_price_kg > 0)]
    if len(x) < 1000:
        continue
    same = np.log(x.qty_kg).diff().corr(np.log(x.avg_price_kg).diff())
    w = x.resample("W").agg({"qty_kg": "sum", "avg_price_kg": "mean"}).replace(0, np.nan).dropna()
    lead = np.log(w.qty_kg).diff().corr(np.log(w.avg_price_kg).diff().shift(-1))
    rows.append((it, same, lead))
corr = pd.DataFrame(rows, columns=["item", "same_day", "next_week"]).sort_values("same_day")
print(corr.round(2).to_string(index=False))
print("중앙값:", corr[["same_day", "next_week"]].median().round(2).to_dict())
fig, axes = plt.subplots(1, 2, figsize=(11, 0.22 * len(corr) + 1.2), sharey=True)
for ax, col, title in zip(axes, ["same_day", "next_week"], ["같은 날: 물량 변화 ↔ 가격 변화", "주 단위: 이번 주 물량 변화 → 다음 주 가격 변화"]):
    v = corr[col].values
    ax.barh(corr.item, v, color=[SERIES[0] if s < 0 else SERIES[7] for s in v], height=0.6)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-0.6, 0.6)
    ax.set_title(title, fontsize=10)
axes[0].tick_params(axis="y", labelsize=7)
fig.suptitle("가락시장 물량과 가격의 상관 (로그 차분, 품목별)", fontsize=11, color=INK2)
save(fig, "05_trade_price_corr")

# %% [markdown]
# ## 4. 축산: 가축질병·사육두수

# %%
d = rd("livestock/disease_events.parquet")
d = d[d.date >= "2016-01-01"]
d["year"] = d.date.dt.year
major = {"고병원성조류인플루엔자": "고병원성 AI", "아프리카돼지열병": "ASF", "구제역": "구제역", "럼피스킨병": "럼피스킨"}
cnt = d[d.disease_std.isin(major)].pivot_table(index="year", columns="disease_std", values="head_count", aggfunc="size", fill_value=0)
print(cnt)
fig, axes = plt.subplots(1, len(major), figsize=(12, 3), sharex=True)
for ax, (k, lab), c in zip(axes, major.items(), SERIES):
    s = cnt.get(k, pd.Series(0, index=cnt.index))
    ax.bar(s.index, s.values, color=c, width=0.7)
    ax.set_title(lab)
    ax.set_xticks([2016, 2019, 2022, 2025])
axes[0].set_ylabel("발생 건수")
fig.suptitle("주요 가축질병 연도별 발생 건수", fontsize=11, color=INK2)
save(fig, "06_disease_by_year")

ce = rd("livestock/census_quarterly.parquet")
ce = ce[(ce["시도별"] == "전국") & (ce.get("사육규모별", "합계").fillna("합계").isin(["합계"]) | ce.get("사육규모별").isna())]
pick = {("livestock_census", "한우:마리수"): "한우", ("chicken_census", "산란계:마리수"): "산란계", ("chicken_census", "육용계:마리수"): "육용계"}
series = {lab: ce[(ce.table == t) & (ce.item_nm == i)].set_index("period").value for (t, i), lab in pick.items()}
pig = pd.concat([ce[(ce.table == t) & (ce.item_nm == "마리수")].set_index("period").value for t in ("pig_census_old", "pig_census")])
series["돼지"] = pig[~pig.index.duplicated(keep="last")]
fig, ax = plt.subplots(figsize=(10, 4))
for c, (lab, s) in zip(SERIES, series.items()):
    s = s.sort_index()
    x = pd.PeriodIndex([f"{p[:4]}Q{int(p[4:])}" for p in s.index], freq="Q").to_timestamp()
    ax.plot(x, s.values / s.values[0] * 100, color=c, label=lab)
ax.axhline(100, color=INK2, lw=1, ls=":")
ax.set_title("사육두수 지수 (2016 1분기 = 100, 분기)")
ax.legend(ncol=4, fontsize=8)
save(fig, "07_livestock_census")

# %% [markdown]
# ## 5. 수산: 어업생산·연안 수온

# %%
fp = rd("fishery/production_monthly.parquet")
fp = fp[(fp.item_id == "T01") & (fp["어업별"] == "계")]
fp["year"] = fp.period.str[:4].astype(int)
sp = ["고등어", "살오징어(오징어)", "멸치", "갈치", "꽃게"]
ann = fp[(fp["품종별"].isin(sp)) & (fp.year < 2026)].pivot_table(index="year", columns="품종별", values="value", aggfunc="sum")
print((ann / 1000).round(0))
fig, ax = plt.subplots(figsize=(10, 4))
for c, s in zip(SERIES, sp):
    ax.plot(ann.index, ann[s] / ann[s].iloc[0] * 100, color=c, marker="o", ms=4, label=s)
ax.axhline(100, color=INK2, lw=1, ls=":")
ax.set_title("주요 어종 연간 생산량 지수 (2016 = 100)")
ax.legend(ncol=5, fontsize=8)
save(fig, "08_fishery_production")

ct = rd("fishery/coast_temp_daily.parquet")
ct = ct[ct.water_temp.notna()]
print("연안정지관측 운영 관측소 수:", ct.groupby(ct.date.dt.year).stn_cd.nunique().to_dict(), "→ 수온 피처는 해양기상부이로 대체")

sea = rd("fishery/sea_temp_daily.parquet")
mon = sea.pivot_table(index=sea.date.dt.to_period("M").dt.to_timestamp(), columns="sea", values="water_temp")
fig, ax = plt.subplots(figsize=(10, 3.8))
for c, s_ in zip(SERIES, ["동해", "남해", "서해"]):
    ax.plot(mon.index, mon[s_], color=c, label=s_, lw=1.6)
ax.set_ylabel("℃")
ax.set_title("해역별 월평균 수온 (기상청 해양기상부이 17곳, 2016~2026)")
ax.legend(ncol=3, fontsize=8)
save(fig, "09_sea_temp")

clim = sea.assign(m=sea.date.dt.month, y=sea.date.dt.year)
anom = clim.groupby(["y", "sea"]).water_temp.mean().unstack() - clim[clim.y < 2026].groupby("sea").water_temp.mean()
print("연평균 수온 편차(℃, 2026은 1~9월):")
print(anom.round(2))

# %% [markdown]
# ## 6. 결측·품질 요약

# %%
qual = []
for f in sorted(P.glob("*/*.parquet")):
    if f.stem == "price":
        continue
    d = pd.read_parquet(f)
    na = d.isna().mean()
    qual.append((f"{f.parent.name}/{f.stem}", len(d), ", ".join(f"{k} {v:.0%}" for k, v in na[na > 0.05].items()) or "-"))
print(pd.DataFrame(qual, columns=["file", "rows", "null>5%"]).to_string(index=False))
