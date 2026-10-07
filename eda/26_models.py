# 트리 기반 모델 비교 (같은 피처·같은 전처리): LightGBM 기본 / LightGBM 큰 모델 / XGBoost / CatBoost
# 피처 = 기본 피처 + (인자로 받은) 채택 파생변수. 시드 0. 검증 2023~2024 전체 행
import importlib.util, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
EXTRA = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
DOMS = sys.argv[2].split(",") if len(sys.argv) > 2 else list(dv.cs.rm.NAMES)


def matrix(df, num, cats, tr):
    num = sorted(num)
    X = dv.cs.rm.prep(df, num, tr)
    for c, ok in df.attrs.get("scope", {}).items():
        if c in X: X.loc[~ok, c] = -1.0
    for c in cats:
        X[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    return X


MODELS = {
    "LightGBM 기본 (600, lr .05, 63잎)": lambda: lgb.LGBMRegressor(**{**dv.cs.rm.LGB, "random_state": 0}),
    "LightGBM 큰 모델 (2000, lr .03, 255잎)": lambda: lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=255, min_child_samples=100,
                                                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1),
    "XGBoost (1000, lr .05, 깊이 8)": lambda: xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                                                            min_child_weight=50, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=0, n_jobs=12),
    "CatBoost (1500, lr .08, 깊이 8)": lambda: CatBoostRegressor(iterations=1500, learning_rate=0.08, depth=8, loss_function="RMSE", random_seed=0, verbose=0, thread_count=12),
}
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.cs.build(d)[0]
    if EXTRA.get(d):
        df = dv.add_derived(df, d)
    tr, te = dv.masks(df); y = df.y.values
    num, cats = dv.base_sets(nm); num = num + EXTRA.get(d, [])
    X = matrix(df, num, cats, tr)
    for name, mk in MODELS.items():
        t = time.time(); m = mk()
        if name.startswith("LightGBM"):
            m.fit(X[tr], y[tr], categorical_feature=cats)
        elif name.startswith("CatBoost"):
            Xc = X.copy()
            for c in cats: Xc[c] = Xc[c].astype(str)
            m.fit(Xc[tr], y[tr], cat_features=cats); p = m.predict(Xc[te])
        else:
            m.fit(X[tr], y[tr])
        if not name.startswith("CatBoost"):
            p = m.predict(X[te])
        r = {"분야": nm, "모델": name, "피처 수": len(num) + len(cats), **{k: round(v, 4) for k, v in dv.metrics(y[te], p).items()}, "학습 초": round(time.time() - t)}
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "model_compare.csv", index=False, encoding="utf-8-sig")
    del df, X
