# %% [markdown]
# # 표준 데이터셋(원본 변수) 성능 측정 (향후 28일 평균 가격)
# 입력: data/processed/{domain}/dataset.parquet (원본 변수). 타깃 y = target_mean_28d / 최근 7일 평균 가격 − 1 (단순 변화율)
# 피처: 원본 변수(year·설추석까지 일수 제외) + 주차(woy) + 가격 이력(r7·r28·dev_last 등 8개). 베이스라인 1 = 오늘 가격 유지, 베이스라인 2 = 최근 7일 평균 유지
# 모델: 선형회귀 · XGBoost · LightGBM (랜덤포레스트는 학습이 오래 걸려 제외). 전처리: 결측 = 학습셋 평균, MinMax(학습셋 fit)
# 학습 2016~2022 / 테스트 2023~2024
# 실행: `python eda/08_raw_dataset_eval.py` → eda/08_raw_dataset_eval.md

# %%
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
TAB = ROOT / "eda" / "tables"
H = 28
SPLIT = ("2016-01-01", "2022-12-01", "2023-01-01", "2024-12-31")   # 학습 2016~2022 (타깃이 2022년 안에 끝나도록 12월 초까지) / 테스트 2023~2024
KEYS = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
CAT = ["item_cd", "food_group", "se_nm"]
LIN_CAT = ["item_cd", "se_nm"]                      # food_group은 item_cd로 결정됨 (선형회귀 공선성)
DROP = {"domain", "date", "ctgry_cd", "item_nm", "vrty_nm", "grd_nm", "target_mean_14d", "target_mean_28d",
        "garak_top_origin", "import_month", "catch_month", "census_period", "production_year", "farm_survey_month",   # 메타·상수 컬럼
        "days_to_seollal", "days_to_chuseok",                     # 명절까지 남은 일수는 이번 측정에서 제외
        "year"}                                                   # 트리는 연도를 외삽하지 못함 → 제외
PRICE_LEVEL = ["price_kg", "whsl_price_kg", "mafra_whsl_price_kg", "garak_price_kg", "auction_price_kg",
               "pig_auction_price_kg", "pig_auction_skinned_kg", "pig_auction_rep_kg", "cpi_item"]   # 가격 '수준' 변수 (원/kg 등)
CAL = ["woy"]                                      # 날짜 가공은 주차만 추가 (연중일은 과적합 우려로 제외, 연도는 원본에 포함)
HIST = ["r7", "r28", "r91", "yoy", "vol28", "dev_last", "ly_change", "n_obs28"]
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}


def rmean(s, lo, hi, minp):
    return s.shift(lo).rolling(hi - lo + 1, min_periods=minp).mean()


def add_base_and_history(df):
    """시계열별 최근 7일 평균(base), 타깃 y, 가격 이력 8개 (과거 값만 사용)."""
    parts = []
    for _, g in df.groupby(KEYS, observed=True, sort=False):
        s = g.set_index("date").price_kg.asfreq("D")
        base = rmean(s, 0, 6, 3)
        f = pd.DataFrame(index=s.index)
        f["base"] = base
        f["r7"] = np.log(base / rmean(s, 7, 13, 3))
        f["r28"] = np.log(base / rmean(s, 28, 34, 3))
        f["r91"] = np.log(base / rmean(s, 91, 97, 3))
        f["yoy"] = np.log(base / rmean(s, 358, 371, 3))
        f["vol28"] = np.log(s).diff().rolling(28, min_periods=10).std()
        f["dev_last"] = np.log(s / base)
        f["ly_change"] = np.log(s.shift(364 - H).rolling(H, min_periods=10).mean() / rmean(s, 358, 364, 3))
        f["n_obs28"] = s.notna().rolling(28, min_periods=1).sum()
        parts.append(f.reindex(g.date).set_index(g.index))
    out = df.join(pd.concat(parts))
    out["y"] = out.target_mean_28d / out.base - 1
    return out


def add_calendar(df):
    df["woy"] = df.date.dt.isocalendar().week.astype(int)                 # ISO 주차 (1~53)
    return df


def metrics(y, yhat):
    return {"RMSE": np.sqrt(np.mean((yhat - y) ** 2)), "MAE": np.mean(np.abs(yhat - y)),
            "R2": 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)}


def fit_eval(df, num, split):
    t0, t1, s0, s1 = split
    ok = df.y.notna() & np.isfinite(df.y)
    tr_m = ok & (df.date >= t0) & (df.date <= t1)
    te_m = ok & (df.date >= s0) & (df.date <= s1)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    X = df[num].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr_m].mean()).fillna(0)                       # 결측 = 학습셋 평균 (학습셋에서도 전부 결측이면 0)
    Xs = pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X), columns=num, index=df.index).astype("float32")
    y = df.y.values
    out = {"베이스라인 1 (오늘 가격 = 향후 4주 평균)": metrics(y[te], np.expm1(df.dev_last.fillna(0).values[te])),
           "베이스라인 2 (최근 7일 평균 = 향후 4주 평균)": metrics(y[te], np.zeros(len(te)))}

    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(df.loc[tr_m, LIN_CAT].astype(str))
    lx = lambda r: hstack([csr_matrix(Xs.values[r]), oh.transform(df.iloc[r][LIN_CAT].astype(str))]).tocsr()
    out["선형회귀"] = metrics(y[te], LinearRegression().fit(lx(tr), y[tr]).predict(lx(te)))

    Xc = Xs.copy()
    for c in CAT:
        Xc[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    xg = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                          tree_method="hist", device="cuda", enable_categorical=True, random_state=0)
    out["XGBoost"] = metrics(y[te], xg.fit(Xc.iloc[tr], y[tr]).predict(Xc.iloc[te]))
    lg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=0, verbose=-1)
    lg.fit(Xc.iloc[tr], y[tr], categorical_feature=CAT)
    out["LightGBM"] = metrics(y[te], lg.predict(Xc.iloc[te]))
    imp = pd.Series(lg.booster_.feature_importance("gain"), index=Xc.columns)
    return out, imp, len(tr), len(te)


def main():
    t_all = time.time()
    rows, imps, info = [], {}, []
    for d, nm in NAMES.items():
        df = pd.read_parquet(P / d / "dataset.parquet")
        df = add_calendar(add_base_and_history(df)).reset_index(drop=True)
        raw = [c for c in df.columns if c not in DROP and c not in CAT and c not in CAL + HIST + ["base", "y"]
               and pd.api.types.is_numeric_dtype(df[c])]
        cols = raw + CAL + HIST
        res, imp, ntr, nte = fit_eval(df, cols, SPLIT)
        for model, mt in res.items():
            rows.append({"분야": nm, "모델": model, **mt})
        imps[nm] = imp
        info.append((nm, len(cols) + len(CAT), ntr, nte))
        print(f"{nm}: 피처 {len(cols) + len(CAT)}, 학습 {ntr:,} / 테스트 {nte:,}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(TAB / "raw_dataset_eval.csv", encoding="utf-8-sig", index=False)

    lines = ["# 표준 데이터셋 성능 (향후 28일 평균 가격, 테스트 2023~2024)", "",
             "> 자동 생성: `python eda/08_raw_dataset_eval.py`",
             "> 타깃 = 향후 28일 평균 가격 / 최근 7일 평균 − 1 (0.05 = 5% 상승). 학습 2016~2022, 테스트 2023~2024",
             "> 베이스라인 1 = 오늘 가격이 향후 4주 평균이라고 예측(직전 가격), 베이스라인 2 = 최근 7일 평균이 그대로 유지된다고 예측(변화율 0). 피처 = 원본 변수(year·설추석까지 일수 제외) + 주차 + 가격 이력 8개",
             "> 전처리: 결측 = 학습셋 평균, MinMax(학습셋 fit)", ""]
    for nm in NAMES.values():
        x = r[r.분야 == nm].drop(columns="분야")
        lines += [f"## {nm}", "", x.round(4).to_markdown(index=False, disable_numparse=True), ""]
    lines += ["## LightGBM 중요 피처 (gain 상위 15)", ""]
    for nm, imp in imps.items():
        imp = imp / imp.sum()
        lines.append(f"- **{nm}**: " + ", ".join(f"{k} ({v:.3f})" for k, v in imp.sort_values(ascending=False).head(15).items()))
    lines += ["", pd.DataFrame(info, columns=["분야", "피처 수", "학습 행", "테스트 행"]).to_markdown(index=False), "",
              f"> 전체 소요 {time.time() - t_all:.0f}초"]
    (ROOT / "eda" / "08_raw_dataset_eval.md").write_text(chr(10).join(lines), encoding="utf-8")
    for nm in NAMES.values():
        print(f"[{nm}]")
        print(r[r.분야 == nm].drop(columns="분야").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
