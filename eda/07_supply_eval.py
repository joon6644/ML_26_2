# %% [markdown]
# # 공급 피처 추가 전후 비교 (향후 30일 평균 가격)
# 피처 세트: 기본(03의 피처) vs 기본 + 공급(supply_features.py). 모델: 선형회귀 · 랜덤포레스트 · XGBoost · LightGBM.
# 분할 A: 학습 2016~2022 / 테스트 2023~2024,  분할 B: 학습 ~2025-08 / 테스트 2025-10 ~ 2026-08
# 전처리(모든 모델 공통): 결측 = 학습셋 평균, MinMax(학습셋 fit). 실행: `python eda/07_supply_eval.py`

# %%
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eda"))
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("m30", ROOT / "eda" / "05_models_h30_2023_24.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
bm = m.bm
import supply_features as sf                                       # noqa: E402
import lightgbm as lgb                                             # noqa: E402
import xgboost as xgb                                              # noqa: E402
from scipy.sparse import csr_matrix, hstack                        # noqa: E402
from sklearn.ensemble import RandomForestRegressor                 # noqa: E402
from sklearn.linear_model import LinearRegression                  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder      # noqa: E402

SPLITS = {"A": ("2016-01-01", "2022-12-01", None, "2023-01-01", "2024-12-31"),
          "B": ("2016-01-01", "2025-08-30", None, "2025-10-01", "2026-08-30")}   # 학습 시작, 학습 끝, (미사용), 테스트 시작, 테스트 끝
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
TAB = ROOT / "eda" / "tables"


def build(domain, common, item_map):
    daily, keys = bm.build_series(domain)
    df = bm.make_rows(daily, keys).merge(common, on="date", how="left").merge(item_map, on=["ctgry_cd", "item_cd"], how="left")
    fn, dom_cols = bm.DOMAIN_FEATURES[domain]
    df = fn(df)
    sfn, sup_cols = sf.SUPPLY[domain]
    df = sfn(df, daily).replace([np.inf, -np.inf], np.nan)
    df["sid"] = df.groupby(keys, observed=True).ngroup()
    df = df.sort_values(["sid", "date"]).reset_index(drop=True)
    return df, bm.NUM + dom_cols, sup_cols


def scaled(df, cols, tr_mask):
    X = df[cols].fillna(df.loc[tr_mask, cols].mean())                  # 학습셋 평균으로 결측 채움 (전부 결측이면 0)
    X = X.fillna(0)
    return pd.DataFrame(MinMaxScaler().fit(X[tr_mask]).transform(X), columns=cols, index=df.index).astype("float32")


def fit_eval(df, num, split):
    t_start, t_end, v_start, s_start, s_end = split
    ok = df.y.notna() & df.base.notna()
    tr_m = ok & (df.date >= t_start) & (df.date <= t_end)
    te_m = ok & (df.date >= s_start) & (df.date <= s_end)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    Xs = scaled(df, num, tr_m)
    y = df.y.values.astype("float32")
    preds = {"기준: 오늘 가격 유지": df.dev_last.fillna(0).values[te]}

    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(df.loc[tr, bm.LIN_CAT].astype(str))
    lx = lambda r: hstack([csr_matrix(Xs.values[r]), oh.transform(df.loc[r, bm.LIN_CAT].astype(str))]).tocsr()
    preds["선형회귀"] = LinearRegression().fit(lx(tr), y[tr]).predict(lx(te))

    Xx = Xs.copy()
    for c in bm.CAT:
        Xx[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    xg = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                          tree_method="hist", device=m.DEV, enable_categorical=True, random_state=0)
    xg.fit(Xx.iloc[tr], y[tr])
    preds["XGBoost"] = xg.predict(Xx.iloc[te])

    # 랜덤포레스트: 범주형은 정수 코드로 (트리 분할용). 학습 속도를 위해 트리마다 학습행의 30%만 샘플링
    Xr = Xs.copy()
    for c in bm.CAT:
        Xr[c] = Xx[c].cat.codes.astype("float32")
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, max_samples=0.3, n_jobs=-1, random_state=0)
    rf.fit(Xr.values[tr], y[tr])
    preds["랜덤포레스트"] = rf.predict(Xr.values[te])

    lg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=0, verbose=-1)
    lg.fit(Xx.iloc[tr], y[tr], categorical_feature=bm.CAT)
    preds["LightGBM"] = lg.predict(Xx.iloc[te])

    res = pd.DataFrame({k: m.metrics(y[te], np.asarray(v, dtype="float64"), df.base.values[te]) for k, v in preds.items()}).T
    imp = pd.Series(xg.get_booster().get_score(importance_type="gain"))
    return res, imp, len(tr), len(te)


def main():
    t0 = time.time()
    common = bm.common_features()
    item_map = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str)[["ctgry_cd", "item_cd", "food_group"]]
    out, imps, miss, info = [], {}, [], []
    for d, nm in NAMES.items():
        t = time.time()
        df, base_cols, sup_cols = build(d, common, item_map)
        print(f"{nm} 피처 생성 {time.time() - t:.0f}초, 행 {len(df):,}", flush=True)
        for sname, split in SPLITS.items():
            trm = (df.date >= split[0]) & (df.date <= split[1]) & df.y.notna()
            tem = (df.date >= split[3]) & (df.date <= split[4]) & df.y.notna()
            miss.append(pd.DataFrame({"도메인": nm, "분할": sname, "피처": sup_cols,
                                      "결측%(학습)": df.loc[trm, sup_cols].isna().mean().values * 100,
                                      "결측%(테스트)": df.loc[tem, sup_cols].isna().mean().values * 100}))
            for fs, cols in (("기본", base_cols), ("기본+공급", base_cols + sup_cols)):
                res, imp, ntr, nte = fit_eval(df, cols, split)
                out.append(res.assign(도메인=nm, 분할=sname, 피처세트=fs).reset_index(names="모델"))
                if fs == "기본+공급":
                    imps[(nm, sname)] = imp
                info.append((nm, sname, fs, ntr, nte, len(cols) + len(bm.CAT)))
                print(f"  {nm} {sname} {fs}: 학습 {ntr:,} / 테스트 {nte:,}", flush=True)
    r = pd.concat(out, ignore_index=True)
    r.to_csv(TAB / "supply_eval.csv", encoding="utf-8-sig", index=False)
    mi = pd.concat(miss, ignore_index=True)
    mi.to_csv(TAB / "supply_missing.csv", encoding="utf-8-sig", index=False)

    lines = ["# 공급 피처 추가 전후 비교 (향후 30일 평균 가격)", "",
             "> 자동 생성: `python eda/07_supply_eval.py` · 지표는 변화율(log) 스케일. 기준 = 오늘 가격 유지",
             "> 분할 A: 학습 2016~2022 / 테스트 2023~2024 · 분할 B: 학습 ~2025-08 / 테스트 2025-10 ~ 2026-08",
             "> 전처리: 결측 = 학습셋 평균, MinMax(학습셋 fit). 공급 피처 정의는 docs/features.md", ""]
    for sname in SPLITS:
        lines += [f"## 분할 {sname}", ""]
        x = r[r.분할 == sname]
        piv = x.pivot_table(index=["도메인", "모델"], columns="피처세트", values=["RMSE", "MAE", "R2"], sort=False)
        piv.columns = [f"{a} ({b})" for a, b in piv.columns]
        lines += [piv.round(4).reset_index().to_markdown(index=False, disable_numparse=True), ""]
    lines += ["## XGBoost 중요 피처 (기본+공급, gain 상위 12)", ""]
    for (nm, sname), imp in imps.items():
        lines.append(f"- **{nm} {sname}**: " + ", ".join(f"{k} ({v:.2f})" for k, v in imp.sort_values(ascending=False).head(12).items()))
    lines += ["", "## 공급 피처 결측률", "", mi.round(1).to_markdown(index=False, disable_numparse=True), "",
              f"> 전체 소요 {time.time() - t0:.0f}초"]
    (ROOT / "eda" / "07_supply_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print(r[["분할", "도메인", "피처세트", "모델", "RMSE", "MAE", "R2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
