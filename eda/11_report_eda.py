# %% [markdown]
# # 1차 보고서용 EDA (데이터 품질 · 가격 구조 · 계절성 · 자기상관 · 이상치 · 외부 변수 · 식재료 군집 · 모델 성능)
# 입력: data/processed/{domain}/dataset.parquet, features.parquet
# 출력: docs/report/figures/*.png, docs/report/eda_results.md (표), eda/tables/report_*.csv
# 실행: `python eda/11_report_eda.py`

# %%
import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "eda" / "tables"
spec = importlib.util.spec_from_file_location("ev", ROOT / "eda" / "08_raw_dataset_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9})
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
COLOR = {"농산물": "#3a8a4f", "축산물": "#c0504d", "수산물": "#2f6fb0"}
KEYS = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
REP = {"agri": ["배추", "양파"], "livestock": ["돼지", "계란"], "fishery": ["고등어", "갈치"]}
OUT = {}            # 보고서에 들어갈 수치
MD = ["# 1차 보고서 EDA 결과 (자동 생성)", "", "> `python eda/11_report_eda.py`", ""]


def table(title, df, note=None):
    MD.extend([f"## {title}", ""] + ([f"_{note}_", ""] if note else []) + [df.to_markdown(index=False, disable_numparse=True), ""])
    df.to_csv(TAB / f"report_{title.split()[0]}.csv", encoding="utf-8-sig", index=False)


def series_daily(g):
    return g.set_index("date").price_kg.asfreq("D")


# %% 데이터 로드
raw = {d: pd.read_parquet(P / d / "dataset.parquet") for d in NAMES}
feat = {d: pd.read_parquet(P / d / "features.parquet") for d in NAMES}
for d in NAMES:
    raw[d] = raw[d][raw[d].date >= "2016-01-01"]      # 2015년은 준비 기간

# %% ① 데이터 구조 및 품질
rows = []
for d, nm in NAMES.items():
    df = raw[d]
    dup = df.duplicated(KEYS + ["date"]).sum()
    n_series = df.groupby(KEYS, observed=True).ngroups
    span = df.groupby(KEYS, observed=True).date.agg(["min", "max", "count"])
    days = (span["max"] - span["min"]).dt.days + 1
    rows.append({"분야": nm, "행 수": f"{len(df):,}", "품목 수": df.item_nm.nunique(), "시계열 수": n_series,
                 "기간": f"{df.date.min():%Y-%m-%d} ~ {df.date.max():%Y-%m-%d}",
                 "소매/중도매": "/".join(sorted(df.se_nm.unique())),
                 "시계열당 조사일 중앙값": int(span["count"].median()),
                 "조사 밀도(조사일/전체일) 중앙값": f"{(span['count'] / days).median():.2f}",
                 "2016년부터 있는 시계열 %": f"{(span['min'] < '2016-02-01').mean() * 100:.0f}",
                 "키+날짜 중복": int(dup), "가격 결측": int(df.price_kg.isna().sum())})
table("01 데이터 구조", pd.DataFrame(rows), "가격 단위는 모두 원/kg로 환산 (개·마리·포기 등은 KAMIS 표준 중량으로 환산)")

GROUP = {  # 변수 그룹 (features.parquet 기준)
    "가격 이력": ev.HIST, "가격·도매가": ["price_kg", "whsl_price_kg", "mafra_whsl_price_kg", "garak_price_kg", "auction_price_kg",
                                     "pig_auction_price_kg", "pig_auction_skinned_kg", "pig_auction_rep_kg"],
    "수급(거래·생산·교역)": ["garak_qty_kg", "garak_amount_krw", "garak_n_trades", "garak_top_origin_share", "production_ton",
                      "cultivated_area_ha", "import_kg", "import_usd", "export_kg", "census_heads", "census_farms",
                      "auction_heads", "stock_ton", "catch_ton", "catch_value_kkrw"],
    "기상": [c for c in feat["agri"].columns if c.startswith(("wx_", "prodarea_"))],
    "해양환경": [c for c in feat["fishery"].columns if c.startswith(("sea_temp", "buoy_", "coast_", "redtide", "jellyfish", "farm"))],
    "가축질병": ["disease_cases", "disease_heads", "disease_all_cases", "disease_all_heads"],
    "경제(물가·환율·원자재)": [c for c in set().union(*[f.columns for f in feat.values()])
                        if c.startswith(("cpi_", "intl_", "fuel_", "impidx_")) or c == "usd_krw"],
    "달력·규제": ["woy", "dow", "is_holiday", "closed_season", "closed_window", "days_to_closed_start"],
    "품목 범주": ["item_cd", "food_group", "se_nm"],
}
G_OF = {c: g for g, cs in GROUP.items() for c in cs}
miss = []
for d, nm in NAMES.items():
    f = feat[d]
    f = f[f.date >= "2016-01-01"]
    for g, cs in GROUP.items():
        cs = [c for c in cs if c in f.columns]
        if cs:
            miss.append({"분야": nm, "그룹": g, "변수 수": len(cs), "결측 %": f[cs].isna().mean().mean() * 100})
miss = pd.DataFrame(miss)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
groups = [g for g in GROUP if g in miss.그룹.values]
yy = np.arange(len(groups))
for i, nm in enumerate(NAMES.values()):
    m = miss[miss.분야 == nm].set_index("그룹").reindex(groups)["결측 %"]
    ax.barh(yy + (i - 1) * 0.26, m.values, height=0.24, color=COLOR[nm], label=nm)
ax.set_yticks(yy, groups)
ax.invert_yaxis()
ax.set_xlabel("평균 결측률 (%)")
ax.set_title("변수 그룹별 평균 결측률 (2016~, 학습용 피처)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "01_missing_by_group.png")
plt.close(fig)
table("02 그룹별 결측률", miss.round(1).astype({"결측 %": str}))

# %% ② 가격 분포
dist, cv_rows = [], []
for d, nm in NAMES.items():
    df = raw[d][raw[d].se_nm == "소매"]
    s = df.groupby(KEYS, observed=True).price_kg
    m = s.mean()
    cv = (s.std() / s.mean()).dropna()
    dist.append(pd.DataFrame({"분야": nm, "평균가": m.values}))
    q = m.quantile([0, .25, .5, .75, 1]).values
    cv_rows.append({"분야": nm, "시계열 수(소매)": len(m), "평균가 최소": f"{q[0]:,.0f}", "25%": f"{q[1]:,.0f}",
                    "중앙값": f"{q[2]:,.0f}", "75%": f"{q[3]:,.0f}", "최대": f"{q[4]:,.0f}",
                    "최대/최소 배율": f"{q[4] / q[0]:,.0f}", "변동계수(CV) 중앙값": f"{cv.median():.2f}"})
table("03 가격 분포", pd.DataFrame(cv_rows), "소매 시계열별 기간 평균 가격(원/kg) 분포, CV = 시계열 내 표준편차/평균")
dist = pd.concat(dist)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].boxplot([np.log10(dist[dist.분야 == nm].평균가) for nm in NAMES.values()], tick_labels=list(NAMES.values()),
                patch_artist=True, boxprops=dict(facecolor="#eeeeee"), medianprops=dict(color="black"))
axes[0].set_ylabel("log10(평균 가격 원/kg)")
axes[0].set_title("시계열별 평균 가격 (소매)")
for nm in NAMES.values():
    r = []
    for d2, n2 in NAMES.items():
        if n2 == nm:
            r = feat[d2].y.clip(-0.5, 0.8)
    axes[1].hist(r, bins=80, density=True, histtype="step", lw=1.5, color=COLOR[nm], label=nm)
axes[1].set_xlabel("타깃 y (향후 4주 평균 / 최근 7일 평균 - 1)")
axes[1].set_title("타깃 분포")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "02_price_distribution.png")
plt.close(fig)

ys = []
for d, nm in NAMES.items():
    y = feat[d].y
    ys.append({"분야": nm, "행 수": f"{len(y):,}", "평균": f"{y.mean():+.3f}", "표준편차": f"{y.std():.3f}",
               "5%": f"{y.quantile(.05):+.3f}", "중앙값": f"{y.median():+.3f}", "95%": f"{y.quantile(.95):+.3f}",
               "|y|>10% 비율": f"{(y.abs() > .1).mean() * 100:.0f}%"})
table("04 타깃 분포", pd.DataFrame(ys))


# %% ③ 시간 패턴 (대표 품목)
def rep_series(d, item):
    df = raw[d][(raw[d].item_nm == item) & (raw[d].se_nm == "소매")]
    k = df.groupby(["vrty_nm", "grd_nm"]).size().idxmax()
    g = df[(df.vrty_nm == k[0]) & (df.grd_nm == k[1])]
    return series_daily(g), f"{item}({k[0]})" if k[0] not in ("", item) else item


fig, axes = plt.subplots(3, 2, figsize=(9, 6.5), sharex=True)
for r, (d, nm) in enumerate(NAMES.items()):
    for c, item in enumerate(REP[d]):
        s, lab = rep_series(d, item)
        ax = axes[r, c]
        ax.plot(s.index, s.values, lw=0.5, color="#bbbbbb", label="일별")
        ax.plot(s.index, s.rolling(28, min_periods=10).mean(), lw=1.4, color=COLOR[nm], label="28일 이동평균")
        ax.set_title(f"{nm} · {lab}", fontsize=9)
        ax.set_ylabel("원/kg")
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=8)
fig.suptitle("대표 품목 소매가격 추이 (2016~2026)")
fig.tight_layout()
fig.savefig(FIG / "03_rep_timeseries.png")
plt.close(fig)

# %% ④ 계절성
seas_rows, prof_items = [], {}
for d, nm in NAMES.items():
    df = raw[d][raw[d].se_nm == "소매"].copy()
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    amps, cons = [], []
    for key, g in df.groupby(KEYS, observed=True):
        mm = g.groupby(["year", "month"]).price_kg.mean().unstack()
        mm = mm[mm.notna().sum(axis=1) >= 10]
        if len(mm) < 3:
            continue
        idx = mm.div(mm.mean(axis=1), axis=0)          # 연평균 대비 월 지수
        prof = idx.mean()
        amps.append(prof.max() - prof.min())
        c = idx.T.corr().values[np.triu_indices(len(idx), 1)]
        cons.append(np.nanmean(c))
        if key[2] in REP[d] or key[2] in ("사과", "무", "닭", "물오징어", "배추", "양파"):
            cover = prof.notna().sum() * 100 + mm.notna().sum().sum() / 100        # 12개월이 다 차 있고 기간이 긴 시계열 우선
            if prof.notna().sum() == 12 and cover > prof_items.get(key[2], (0, None))[0]:
                prof_items[key[2]] = (cover, prof)
    seas_rows.append({"분야": nm, "분석 시계열": len(amps), "월 지수 진폭 중앙값": f"{np.median(amps):.2f}",
                      "진폭 ≥ 0.3 비율": f"{np.mean(np.array(amps) >= .3) * 100:.0f}%",
                      "연도 간 계절 패턴 상관 중앙값": f"{np.nanmedian(cons):.2f}"})
table("05 계절성", pd.DataFrame(seas_rows),
      "월 지수 = 월평균/그해 평균. 진폭 = 최고 월 − 최저 월 (0.3 = 연평균 대비 30%p 차이). 연도 간 상관 = 해마다 월 패턴이 얼마나 반복되는지")
order = ["배추", "무", "양파", "사과", "돼지", "계란", "닭", "고등어", "물오징어", "갈치"]
H = pd.DataFrame({k: prof_items[k][1] for k in order if k in prof_items}).T
fig, ax = plt.subplots(figsize=(7.5, 3.6))
im = ax.imshow(H.values, cmap="RdBu_r", vmin=0.6, vmax=1.4, aspect="auto")
ax.set_xticks(range(12), [f"{m}월" for m in range(1, 13)])
ax.set_yticks(range(len(H)), H.index)
ax.grid(False)
for i in range(H.shape[0]):
    for j in range(12):
        ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(H.values[i, j] - 1) > 0.25 else "black")
fig.colorbar(im, ax=ax, label="연평균 대비 월 지수")
ax.set_title("품목별 월별 가격 지수 (2016~2026 평균, 1.0 = 그해 평균)")
fig.tight_layout()
fig.savefig(FIG / "04_seasonality_heatmap.png")
plt.close(fig)

dow = []
for d, nm in NAMES.items():
    df = raw[d][raw[d].se_nm == "소매"].copy()
    df["lp"] = np.log(df.price_kg) - df.groupby(KEYS, observed=True).price_kg.transform(lambda s: np.log(s).rolling(7, min_periods=3, center=True).mean())
    df["dow"] = df.date.dt.dayofweek
    v = df.groupby("dow").lp.mean() * 100
    dow.append({"분야": nm, **{"월화수목금토일"[i]: (f"{v[i]:+.2f}" if i in v.index and pd.notna(v[i]) else "조사 없음") for i in range(7)}})
table("06 요일 효과", pd.DataFrame(dow), "주변 7일 평균 대비 요일별 가격 차이(%). 조사는 주로 평일")

# %% ⑤ 자기상관
ac_rows = []
for d, nm in NAMES.items():
    f = feat[d]
    res = {k: [] for k in ["p1", "p7", "p28", "p91", "c28", "c364"]}
    for _, g in raw[d].groupby(KEYS, observed=True):
        s = np.log(series_daily(g))
        if s.notna().sum() < 400:
            continue
        for lag in (1, 7, 28, 91):
            res[f"p{lag}"].append(s.corr(s.shift(lag)))
        ch = s.rolling(7, min_periods=3).mean().shift(-28) - s.rolling(7, min_periods=3).mean()   # 4주 변화 (log)
        res["c28"].append(ch.corr(ch.shift(28)))
        res["c364"].append(ch.corr(ch.shift(364)))
    ac_rows.append({"분야": nm, "가격 lag1": f"{np.nanmedian(res['p1']):.3f}", "lag7": f"{np.nanmedian(res['p7']):.3f}",
                    "lag28": f"{np.nanmedian(res['p28']):.3f}", "lag91": f"{np.nanmedian(res['p91']):.3f}",
                    "4주 변화 vs 직전 4주 변화": f"{np.nanmedian(res['c28']):+.3f}",
                    "4주 변화 vs 1년 전 같은 시기 변화": f"{np.nanmedian(res['c364']):+.3f}"})
table("07 자기상관", pd.DataFrame(ac_rows),
      "시계열별 상관의 중앙값 (log 가격, 일 단위). 가격 수준은 자기상관이 매우 높지만, 4주 변화율은 직전 변화와 음(−)의 상관(되돌림), 1년 전 같은 시기와 양(+)의 상관(계절 반복)")

# %% ⑧ 이상치 (일간 급변)
out_rows = []
for d, nm in NAMES.items():
    n_obs = n_jump = n_rev = 0
    for _, g in raw[d].groupby(KEYS, observed=True):
        p = g.sort_values("date").price_kg.values
        if len(p) < 3:
            continue
        lp = np.log(p)
        dlt = np.diff(lp)
        n_obs += len(dlt)
        jump = np.where(np.abs(dlt[:-1]) > np.log(1.5))[0]       # 직전 조사일 대비 ±50% 이상
        n_jump += len(jump)
        n_rev += int(np.sum(np.abs(lp[jump + 2] - lp[jump]) < np.log(1.1)))   # 다음 조사일에 원래 수준(±10%)으로 복귀
    out_rows.append({"분야": nm, "일간 변화 수": f"{n_obs:,}", "±50% 이상 급변": f"{n_jump:,}",
                     "급변 비율": f"{n_jump / n_obs * 100:.2f}%", "다음 조사일 원복(오류 의심)": f"{n_rev:,}",
                     "원복 비율": f"{n_rev / max(n_jump, 1) * 100:.0f}%"})
table("08 이상치", pd.DataFrame(out_rows),
      "급변 = 직전 조사일 대비 ±50% 이상. 다음 조사일에 원래 수준 ±10%로 돌아오면 입력 오류·일시적 튐 의심, 나머지는 실제 시장 이벤트 후보")

# %% ⑦ 외부 변수: 그룹 SHAP + 가격만 vs 외부 변수 포함
IDS = ["domain", "date", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm", "food_group", "base", "y"]
abl, gshap = [], []
for d, nm in NAMES.items():
    f = feat[d]
    allnum = [c for c in f.columns if c not in IDS]
    price_only = ev.HIST + ["woy", "price_kg"]
    for lab, cols in [("가격 정보만 (가격 이력 + 주차 + 오늘 가격 + 품목)", price_only), ("+ 외부 변수 전체", allnum)]:
        res, imp, _, _ = ev.fit_eval(f, cols, ev.SPLIT)
        for model in ("XGBoost", "LightGBM"):
            abl.append({"분야": nm, "피처": lab, "모델": model, **{k: round(v, 4) for k, v in res[model].items()}})
    # 그룹 SHAP (LightGBM, 테스트 2만 행)
    t0, t1, s0, s1 = ev.SPLIT
    ok = f.y.notna()
    tr_m, te_m = ok & (f.date >= t0) & (f.date <= t1), ok & (f.date >= s0) & (f.date <= s1)
    X = f[allnum].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr_m].mean()).fillna(0)
    Xc = pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X), columns=allnum, index=f.index).astype("float32")
    for c in ev.CAT:
        Xc[c] = f[c].astype(str).astype(pd.CategoricalDtype(sorted(f[c].astype(str).unique())))
    lg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    lg.fit(Xc[tr_m], f.y[tr_m], categorical_feature=ev.CAT)
    te = np.random.default_rng(0).choice(np.where(te_m)[0], min(20000, te_m.sum()), replace=False)
    sv = np.abs(lg.predict(Xc.iloc[te], pred_contrib=True)[:, :-1]).mean(0)
    share = pd.Series(sv / sv.sum(), index=Xc.columns)
    top = share.sort_values(ascending=False).head(8)
    OUT[f"top_{d}"] = [(k, round(v * 100, 1)) for k, v in top.items()]
    for g, v in share.groupby(lambda c: G_OF.get(c, "기타")).sum().items():
        gshap.append({"분야": nm, "그룹": g, "SHAP 비중 %": v * 100})
abl = pd.DataFrame(abl)
table("09 외부 변수 효과", abl, "학습 2016~2022 / 테스트 2023~2024. 품목 범주(item_cd·food_group·se_nm)는 두 경우 모두 포함")
gshap = pd.DataFrame(gshap)
table("10 그룹 SHAP", gshap.round(1).astype({"SHAP 비중 %": str}))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
groups = [g for g in GROUP if g in gshap.그룹.values]
yy = np.arange(len(groups))
for i, nm in enumerate(NAMES.values()):
    m = gshap[gshap.분야 == nm].set_index("그룹").reindex(groups)["SHAP 비중 %"].fillna(0)
    ax.barh(yy + (i - 1) * 0.26, m.values, height=0.24, color=COLOR[nm], label=nm)
ax.set_yticks(yy, groups)
ax.invert_yaxis()
ax.set_xlabel("SHAP 기여 비중 (%)")
ax.set_title("변수 그룹별 예측 기여도 (LightGBM, 테스트 2023~2024)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "05_group_shap.png")
plt.close(fig)

# %% ⑨⑩ 식재료 특성 · 군집
NUT = ["nutr_kcal", "nutr_protein_g", "nutr_fat_g", "nutr_carb_g", "nutr_sugar_g", "nutr_fiber_g", "nutr_sodium_mg"]
items = []
for d, nm in NAMES.items():
    df = pd.read_parquet(P / d / "dataset.parquet", columns=["item_nm", "food_group"] + [c for c in NUT if c in raw[d].columns])
    items.append(df.groupby("item_nm").first().assign(분야=nm))
items = pd.concat(items)
for c in NUT:
    if c not in items:
        items[c] = 0.0
items[NUT] = items[NUT].fillna(0)
OUT["n_items_nutr"] = int(len(items))
Z = StandardScaler().fit_transform(np.log1p(items[NUT].clip(lower=0)))
sil = {k: silhouette_score(Z, KMeans(k, n_init=20, random_state=0).fit_predict(Z)) for k in range(3, 13)}
k_best = max(sil, key=sil.get)
lab = KMeans(k_best, n_init=20, random_state=0).fit_predict(Z)
items["군집"] = lab
OUT["k_best"], OUT["sil_best"] = k_best, round(sil[k_best], 3)
OUT["ari_food_group"] = round(adjusted_rand_score(items.food_group, lab), 3)
OUT["ari_domain"] = round(adjusted_rand_score(items.분야, lab), 3)
pca = PCA(2).fit(Z)
pc = pca.transform(Z)
OUT["pca_var"] = [round(v * 100, 1) for v in pca.explained_variance_ratio_]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.3, 1]})
ax = axes[0]
for nm in NAMES.values():
    m = (items.분야 == nm).values
    ax.scatter(pc[m, 0], pc[m, 1], s=22, color=COLOR[nm], label=nm, edgecolor="white", linewidth=0.6)
show = ["배추", "양파", "사과", "쌀", "콩", "감자", "돼지", "소", "계란", "우유", "고등어", "물오징어", "새우", "김",
        "마른멸치", "천일염", "아몬드", "바나나", "건고추", "새송이버섯", "닭", "갈치", "전복", "고구마"]
for i, name in enumerate(items.index):
    if name in show:
        ax.annotate(name, pc[i], fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.set_xlabel(f"PC1 ({OUT['pca_var'][0]}%)")
ax.set_ylabel(f"PC2 ({OUT['pca_var'][1]}%)")
ax.set_title("영양성분 기반 식재료 분포 (PCA)")
ax.legend(frameon=False, fontsize=8)
axes[1].plot(list(sil), list(sil.values()), marker="o", color="#555555")
axes[1].axvline(k_best, color="#c0504d", ls="--", lw=1)
axes[1].set_xlabel("군집 수 k")
axes[1].set_ylabel("실루엣 계수")
axes[1].set_title("K-Means 군집 수별 실루엣")
fig.tight_layout()
fig.savefig(FIG / "06_nutrition_pca.png")
plt.close(fig)
cl = []
for k, g in items.groupby("군집"):
    prof = g[NUT].median()
    cl.append({"군집": k + 1, "품목 수": len(g),
               "특징 (중앙값 kcal / 단백질 / 지방 / 탄수 g)": f"{prof.nutr_kcal:.0f} / {prof.nutr_protein_g:.1f} / {prof.nutr_fat_g:.1f} / {prof.nutr_carb_g:.1f}",
               "품목 예": ", ".join(g.index[:10]) + (" …" if len(g) > 10 else "")})
table("11 영양 군집", pd.DataFrame(cl), f"K-Means k={k_best} (실루엣 최대 {OUT['sil_best']}), 영양성분 7개 log1p + 표준화")

# %% 모델 성능 (확정 포맷)
perf = []
for d, nm in NAMES.items():
    f = feat[d]
    num = [c for c in f.columns if c not in IDS]
    res, _, ntr, nte = ev.fit_eval(f, num, ev.SPLIT)
    OUT[f"n_{d}"] = (ntr, nte)
    for model, mt in res.items():
        perf.append({"분야": nm, "모델": model, **{k: round(v, 4) for k, v in mt.items()}})
perf = pd.DataFrame(perf)
table("12 모델 성능", perf, "학습 2016~2022 / 테스트 2023~2024, features.parquet 전체 피처")
(ROOT / "docs" / "report" / "eda_results.md").write_text("\n".join(MD), encoding="utf-8")
(ROOT / "docs" / "report" / "eda_numbers.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("\n".join(MD))
print(json.dumps(OUT, ensure_ascii=False, default=str))
