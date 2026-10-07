# %% [markdown]
# # SHAP 기여도 (표준 데이터셋 원본 변수, 향후 28일 평균 가격, 단순 변화율 타깃)
# 08의 ③ 세트(원본 + 주차 + 가격 이력, year·설추석까지 일수 제외)로 LightGBM·XGBoost를 학습하고
# 테스트셋에서 TreeSHAP 평균 |SHAP|을 계산한다. 테스트 2023~2024 × 두 모델 = 2회. 비중 = 피처 |SHAP| / 전체 합
# 실행: `python eda/09_shap.py` → eda/09_shap.md, eda/tables/shap_share.csv

# %%
import importlib.util
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ev", ROOT / "eda" / "08_raw_dataset_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
N_SHAP = 20000


def shap_run(df, num, split):
    t0, t1, s0, s1 = split
    ok = df.y.notna() & np.isfinite(df.y)
    tr_m = ok & (df.date >= t0) & (df.date <= t1)
    te_m = ok & (df.date >= s0) & (df.date <= s1)
    X = df[num].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr_m].mean()).fillna(0)
    Xc = pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X), columns=num, index=df.index).astype("float32")
    for c in ev.CAT:
        Xc[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    te_s = np.random.default_rng(0).choice(te, min(N_SHAP, len(te)), replace=False)
    y = df.y.values
    out = {}
    lg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=0, verbose=-1)
    lg.fit(Xc.iloc[tr], y[tr], categorical_feature=ev.CAT)
    out["LightGBM"] = np.abs(lg.predict(Xc.iloc[te_s], pred_contrib=True)[:, :-1]).mean(0)
    xg = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                          tree_method="hist", device="cuda", enable_categorical=True, random_state=0)
    xg.fit(Xc.iloc[tr], y[tr])
    dm = xgb.DMatrix(Xc.iloc[te_s], enable_categorical=True)
    out["XGBoost"] = np.abs(xg.get_booster().predict(dm, pred_contribs=True)[:, :-1]).mean(0)
    cols = list(Xc.columns)
    return {k: pd.Series(v, index=cols) for k, v in out.items()}


def main():
    rows = []
    for d, nm in ev.NAMES.items():
        df = pd.read_parquet(ev.P / d / "dataset.parquet")
        df = ev.add_calendar(ev.add_base_and_history(df)).reset_index(drop=True)
        raw = [c for c in df.columns if c not in ev.DROP and c not in ev.CAT and c not in ev.CAL + ev.HIST + ["base", "y"]
               and pd.api.types.is_numeric_dtype(df[c])]
        for sname, split in {"2023~24": ev.SPLIT}.items():
            res = shap_run(df, raw + ev.CAL + ev.HIST, split)
            for model, s in res.items():
                share = s / s.sum()
                for f, v in share.items():
                    rows.append({"도메인": nm, "분할": sname, "모델": model, "피처": f, "비중": v})
            print(f"{nm} {sname} 완료", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(ev.TAB / "shap_share.csv", encoding="utf-8-sig", index=False)
    agg = r.groupby(["도메인", "피처"]).비중.agg(["mean", "max"]).reset_index().sort_values(["도메인", "max"])
    agg.to_csv(ev.TAB / "shap_share_agg.csv", encoding="utf-8-sig", index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
