# %% [markdown]
# # 베이스라인 모델: 향후 14일 평균 가격 (전처리 전 간이 버전)
# 도메인(농·축·수) × 모델(선형회귀, XGBoost) = 6개. 기준 모델(가격 유지, 작년 같은 시기)과 비교.
# 실행: `python eda/03_baseline_models.py` → eda/03_baseline_models.md, eda/tables/baseline_*.csv

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
TAB = ROOT / "eda" / "tables"
import sys

H = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 14   # 예측 기간 (일): python ... 30
DATA_END = pd.Timestamp("2026-09-29")
TRAIN_START = "2016-01-01"
TRAIN_END = str((pd.Timestamp("2025-09-29") - pd.Timedelta(days=H)).date())  # 학습 타깃 창이 테스트 시작 전에 끝나도록
TEST_START, TEST_END = "2025-10-01", str((DATA_END - pd.Timedelta(days=H)).date())
SUFFIX = "" if H == 14 else f"_h{H}"                                            # 14일 외 결과는 파일명에 기간 표시
MAJOR_DISEASE = {"계란": "고병원성조류인플루엔자", "닭": "고병원성조류인플루엔자", "돼지": "아프리카돼지열병",
                 "수입 돼지고기": "아프리카돼지열병", "소": "구제역", "수입 소고기": "구제역", "우유": "구제역"}


def rmean(s, lo, hi, minp):
    """t 기준 [t-hi, t-lo] 창의 평균 (lo<=hi, 과거만). 관측 수가 minp 미만이면 NaN."""
    return s.shift(lo).rolling(hi - lo + 1, min_periods=minp).mean()


def fmean(s, h, minp):
    """t 기준 [t+1, t+h] 창의 평균 (미래 = 타깃 전용)."""
    return s[::-1].rolling(h, min_periods=minp).mean()[::-1].shift(-1)


# %% [markdown]
# ## 1. 시계열 만들기 (전국 일별 중앙값, 원/kg)

# %%
def build_series(domain):
    p = pd.read_parquet(P / domain / "price.parquet", columns=["date", "se_nm", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "price_kg"])
    p = p[p.se_nm.isin(["소매", "중도매"]) & (p.price_kg > 0)]
    keys = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
    daily = p.groupby(keys + ["date"], observed=True).price_kg.median().reset_index()
    return daily, keys


def make_rows(daily, keys):
    cal = pd.date_range(daily.date.min(), daily.date.max())
    out = []
    for k, g in daily.groupby(keys, observed=True):
        s = g.set_index("date").price_kg.reindex(cal)
        lp = np.log(s)
        base = rmean(s, 0, 6, 3)
        f = pd.DataFrame(index=cal)
        f["base"] = base
        f["y"] = np.log(fmean(s, H, max(5, H // 3)) / base)                         # 타깃
        f["observed"] = s.notna()
        f["r7"] = np.log(base / rmean(s, 7, 13, 3))
        f["r28"] = np.log(base / rmean(s, 28, 34, 3))
        f["r91"] = np.log(base / rmean(s, 91, 97, 3))
        f["yoy"] = np.log(base / rmean(s, 358, 371, 3))
        f["vol28"] = lp.diff().rolling(28, min_periods=10).std()
        f["dev_last"] = np.log(s / base)
        # 작년 같은 시기의 "향후 14일 변화율" (과거 정보만 사용: t-364 기준 창)
        ly_base = rmean(s, 358, 364, 3)
        ly_fut = s.shift(364 - H).rolling(H, min_periods=max(5, H // 3)).mean()      # [t-364+1, t-364+14]
        f["ly_change"] = np.log(ly_fut / ly_base)
        f["n_obs28"] = s.notna().rolling(28, min_periods=1).sum()
        for c, v in zip(keys, k):
            f[c] = v
        out.append(f[f.observed].drop(columns="observed").reset_index(names="date"))
    return pd.concat(out, ignore_index=True)


# %% [markdown]
# ## 2. 공통·도메인 피처

# %%
def common_features():
    cal = pd.read_parquet(P / "common" / "calendar.parquet").set_index("date")
    w = pd.read_parquet(P / "common" / "weather_national_daily.parquet").set_index("date").asfreq("D")
    fx = pd.read_parquet(P / "common" / "fx_daily.parquet").set_index("date").usd_krw.asfreq("D").ffill(limit=5)
    f = pd.DataFrame(index=pd.date_range("2015-01-01", "2026-09-30"))
    f["month"] = f.index.month
    f["dow"] = f.index.dayofweek
    f["to_seollal"] = cal.days_to_seollal.reindex(f.index).clip(upper=90)
    f["to_chuseok"] = cal.days_to_chuseok.reindex(f.index).clip(upper=90)
    t14 = w.temp_avg.rolling(14, min_periods=10).mean().reindex(f.index)
    clim = w.temp_avg[(w.index >= TRAIN_START) & (w.index <= TRAIN_END)].groupby(w.index[(w.index >= TRAIN_START) & (w.index <= TRAIN_END)].month).mean()
    f["temp14_anom"] = t14 - f.index.month.map(clim).values               # 평년(학습 기간) 대비 기온 편차
    f["rain14"] = w.rain.rolling(14, min_periods=10).sum().reindex(f.index)
    f["heat14"] = (w.temp_max >= 30).rolling(14, min_periods=10).sum().reindex(f.index)
    f["cold14"] = (w.temp_min <= -5).rolling(14, min_periods=10).sum().reindex(f.index)
    f["fx28"] = np.log(fx / fx.shift(28)).reindex(f.index)
    return f.rename_axis("date").reset_index()


def agri_features(df):
    t = pd.read_parquet(P / "agri" / "trade_by_item_daily.parquet")
    parts = []
    for (ic,), g in t.groupby(["item_cd"]):
        q = g.set_index("date").qty_kg.asfreq("D").fillna(0)                 # 휴장일 거래량 = 0
        q = q[q.index >= "2018-01-03"]
        s7, p7 = q.rolling(7).sum(), q.shift(7).rolling(7).sum()
        yq = q.shift(364).rolling(7).sum()
        parts.append(pd.DataFrame({"date": q.index, "item_cd": ic,
                                   "garak_q7_chg": np.log((s7 + 1) / (p7 + 1)).values,
                                   "garak_q7_yoy": np.log((s7 + 1) / (yq + 1)).values}))
    return df.merge(pd.concat(parts), on=["date", "item_cd"], how="left")


def livestock_features(df):
    d = pd.read_parquet(P / "livestock" / "disease_events.parquet")
    idx = pd.date_range("2015-01-01", "2026-09-30")
    cnt = {k: d[d.disease_std == k].groupby("date").size().reindex(idx, fill_value=0).rolling(28).sum() for k in set(MAJOR_DISEASE.values())}
    df = df.copy()
    df["disease28"] = [cnt[MAJOR_DISEASE[i]].get(t, np.nan) for i, t in zip(df.item_nm, df.date)]
    pr = pd.read_parquet(P / "livestock" / "pig_rep_price_daily.parquet")
    pr = pr[pr.skin == "탕박"].groupby("date").price.mean().asfreq("D").ffill(limit=5)
    pig = pd.DataFrame({"date": pr.index, "pig_whsl_7": np.log(pr / pr.shift(7)).values, "pig_whsl_28": np.log(pr / pr.shift(28)).values})
    return df.merge(pig, on="date", how="left")


def fishery_features(df):
    s = pd.read_parquet(P / "fishery" / "sea_temp_daily.parquet").groupby("date").water_temp.mean().asfreq("D")
    s14 = s.rolling(14, min_periods=7).mean()
    tr = s[(s.index >= TRAIN_START) & (s.index <= TRAIN_END)]
    clim = tr.groupby(tr.index.month).mean()
    anom = s14 - s14.index.month.map(clim).values
    return df.merge(pd.DataFrame({"date": anom.index, "sea14_anom": anom.values}), on="date", how="left")


DOMAIN_FEATURES = {"agri": (agri_features, ["garak_q7_chg", "garak_q7_yoy"]),
                   "livestock": (livestock_features, ["disease28", "pig_whsl_7", "pig_whsl_28"]),
                   "fishery": (fishery_features, ["sea14_anom"])}
NUM = ["r7", "r28", "r91", "yoy", "vol28", "dev_last", "ly_change", "n_obs28",
       "month", "dow", "to_seollal", "to_chuseok", "temp14_anom", "rain14", "heat14", "cold14", "fx28"]
CAT = ["item_cd", "food_group", "se_nm"]
LIN_CAT = ["item_cd", "se_nm"]  # food_group은 item_cd로 결정되는 값이라 선형회귀에 같이 넣으면 완전 공선성


# %% [markdown]
# ## 3. 학습·평가

# %%
def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def evaluate(y, yhat, base, fut):
    """변화율(log) 스케일과 가격(원/kg) 스케일에서 MAE·RMSE·R², 그리고 MAPE·방향정확도."""
    err = yhat - y
    price_hat = base * np.exp(yhat)
    perr = price_hat - fut
    moved = np.abs(y) >= 0.01                                               # 1% 이상 움직인 경우만 방향 평가
    return {"MAE(log)": np.mean(np.abs(err)), "RMSE(log)": np.sqrt(np.mean(err ** 2)), "R2(log)": r2(y, yhat),
            "MAE(원/kg)": np.mean(np.abs(perr)), "RMSE(원/kg)": np.sqrt(np.mean(perr ** 2)), "R2(가격)": r2(fut, price_hat),
            "MAPE(%)": np.mean(np.abs(perr) / fut) * 100,
            "방향정확도(%)": np.mean(np.sign(yhat[moved]) == np.sign(y[moved])) * 100 if moved.any() and np.any(yhat != 0) else np.nan}  # 항상 0을 예측하면 방향 없음


def run_domain(domain, common, item_map):
    t0 = time.time()
    daily, keys = build_series(domain)
    df = make_rows(daily, keys)
    df = df.merge(common, on="date", how="left").merge(item_map, on=["ctgry_cd", "item_cd"], how="left")
    fn, dom_cols = DOMAIN_FEATURES[domain]
    df = fn(df)
    num = NUM + dom_cols
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df.y.notna() & df.base.notna()]
    tr = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)]
    te = df[(df.date >= TEST_START) & (df.date <= TEST_END)]
    y_tr, y_te = tr.y.values, te.y.values
    fut = te.base.values * np.exp(y_te)

    res, preds = {}, {}
    preds["기준: 가격 유지"] = np.zeros(len(te))                          # 최근 7일 평균이 유지
    preds["기준: 오늘 가격 유지"] = te.dev_last.fillna(0).values            # 오늘(t) 가격이 유지 (더 강한 기준)
    preds["기준: 작년 같은 시기"] = te.ly_change.fillna(0).values

    lin = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), LIN_CAT),
        ]),
        LinearRegression(),
    )
    lin.fit(tr[num + LIN_CAT], y_tr)
    preds["선형회귀"] = lin.predict(te[num + LIN_CAT])

    Xtr, Xte = tr[num + CAT].copy(), te[num + CAT].copy()
    for c in CAT:
        cats = pd.CategoricalDtype(sorted(df[c].dropna().astype(str).unique()))
        Xtr[c], Xte[c] = Xtr[c].astype(str).astype(cats), Xte[c].astype(str).astype(cats)
    xg = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=20, tree_method="hist", enable_categorical=True, n_jobs=-1, random_state=0)
    xg.fit(Xtr, y_tr)
    preds["XGBoost"] = xg.predict(Xte)

    for name, yhat in preds.items():
        res[name] = evaluate(y_te, np.asarray(yhat), te.base.values, fut)
    out = P / "baseline_preds"                                                  # 테스트 예측값 저장 (git 제외 폴더)
    out.mkdir(exist_ok=True)
    te[["date", "item_nm", "vrty_nm", "grd_nm", "se_nm", "base", "y"]].assign(
        fut=fut, **{f"pred_{k}": np.asarray(v) for k, v in preds.items()}).to_parquet(out / f"{domain}{SUFFIX}.parquet", index=False)
    res = pd.DataFrame(res).T
    naive = res.loc["기준: 가격 유지", "MAE(log)"]
    res["가격유지 대비 개선(%)"] = (1 - res["MAE(log)"] / naive) * 100
    res["오늘가격유지 대비 개선(%)"] = (1 - res["MAE(log)"] / res.loc["기준: 오늘 가격 유지", "MAE(log)"]) * 100

    imp = pd.Series(xg.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    # 품목별 성능 (XGBoost vs 가격 유지)
    te = te.assign(err_x=np.abs(preds["XGBoost"] - y_te), err_n=np.abs(y_te))
    by_item = te.groupby("item_nm").agg(n=("y", "size"), MAE_XGB=("err_x", "mean"), MAE_유지=("err_n", "mean"))
    by_item["개선(%)"] = (1 - by_item.MAE_XGB / by_item.MAE_유지) * 100
    info = {"시계열 수": df.groupby(keys).ngroups, "학습 행": len(tr), "테스트 행": len(te), "피처 수": len(num) + len(CAT),
            "학습 기간": f"{tr.date.min():%Y-%m-%d} ~ {tr.date.max():%Y-%m-%d}", "소요(초)": round(time.time() - t0)}
    print(domain, info)
    return res, imp, by_item.sort_values("n", ascending=False), info


# %%
if __name__ == "__main__":
    common = common_features()
    item_map = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str)[["ctgry_cd", "item_cd", "food_group"]]
    names = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
    lines = [f"# 베이스라인 모델 결과 (향후 {H}일 평균 가격)", "",
             "> 자동 생성: `python eda/03_baseline_models.py` · 전처리·튜닝 전 간이 버전",
             f"> 타깃 y = log(향후 {H}일 평균 / 최근 7일 평균), 시계열 = 품목×품종×등급×소매/중도매의 전국 일별 중앙값(원/kg)",
             f"> 학습 origin {TRAIN_START} ~ {TRAIN_END}, 테스트 origin {TEST_START} ~ {TEST_END}", ""]
    summary = []
    for d in ("agri", "livestock", "fishery"):
        res, imp, by_item, info = run_domain(d, common, item_map)
        res.to_csv(TAB / f"baseline_{d}{SUFFIX}.csv", encoding="utf-8-sig")
        by_item.to_csv(TAB / f"baseline_{d}_by_item{SUFFIX}.csv", encoding="utf-8-sig")
        summary.append(res[["MAE(log)", "RMSE(log)", "R2(log)", "MAE(원/kg)", "RMSE(원/kg)", "R2(가격)", "MAPE(%)", "방향정확도(%)", "오늘가격유지 대비 개선(%)"]].assign(도메인=names[d]))
        lines += [f"## {names[d]}", "", " · ".join(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}" for k, v in info.items()), "",
                  res.round(3).reset_index(names="모델").to_markdown(index=False, disable_numparse=True), "",
                  "**XGBoost 중요 피처 (gain 상위 10)**: " + ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(10).items()), "",
                  "<details><summary>품목별 (테스트 행 많은 순)</summary>", "",
                  by_item.round(3).reset_index().to_markdown(index=False, disable_numparse=True), "", "</details>", ""]
    s = pd.concat(summary).reset_index(names="모델")
    s.to_csv(TAB / f"baseline_summary{SUFFIX}.csv", encoding="utf-8-sig", index=False)
    lines[6:6] = ["## 요약", "", s[["도메인", "모델", "MAE(log)", "RMSE(log)", "R2(log)", "MAE(원/kg)", "RMSE(원/kg)", "R2(가격)", "MAPE(%)", "방향정확도(%)", "오늘가격유지 대비 개선(%)"]].round(4).to_markdown(index=False, disable_numparse=True), ""]
    notes = ROOT / "eda" / f"03_baseline_notes{SUFFIX}.md"                              # 사람이 쓴 해석 (자동 생성에 덮어쓰이지 않게 분리)
    if notes.exists():
        lines[6:6] = [notes.read_text(encoding="utf-8"), ""]
    (ROOT / "eda" / f"03_baseline_models{SUFFIX}.md").write_text("\n".join(lines), encoding="utf-8")
    print(s.round(3).to_string(index=False))
