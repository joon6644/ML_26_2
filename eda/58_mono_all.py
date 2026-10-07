# 단조 제약: (a) 3종 모두 없음 / (b) 3종 모두 적용 / (현재) LGB만. 축·수, 시드 42·0·1·2·3
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
SEEDS = [42, 0, 1, 2, 3]
cls = lambda v: np.where(v > .02, 1, np.where(v < -.02, -1, 0))
rows = []
for d in ["livestock", "fishery"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]; huber = cfg.get("params", {}).get("objective") == "huber"
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    mono = {k: v for k, v in dv.MONO[d].items() if k in num}
    print(nm, "제약 피처", len(mono), mono, flush=True)
    res = {}
    for s in SEEDS:
        X = dv.matrix(df, num, cats, tr, s)
        Xc = X.copy()
        for c in cats: Xc[c] = Xc[c].astype(str)
        cons = [mono.get(c, 0) for c in X.columns]
        P = {}
        for use in (False, True):
            t = time.time()
            pl = {**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s}
            if use: pl.update(monotone_constraints=cons, monotone_constraints_method="advanced")
            P[("lgb", use)] = lgb.LGBMRegressor(**pl).fit(X[tr], yt, categorical_feature=cats).predict(X[te])
            px = dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                      enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=s, **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {}))
            if use: px["monotone_constraints"] = {k: v for k, v in mono.items()}
            P[("xgb", use)] = xgb.XGBRegressor(**px).fit(X[tr], yt).predict(X[te])
            pc = dict(iterations=800, learning_rate=0.1, depth=8, loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=s, verbose=0, thread_count=12)
            if use: pc["monotone_constraints"] = {k: v for k, v in mono.items()}
            P[("cat", use)] = CatBoostRegressor(**pc).fit(Xc[tr], yt, cat_features=cats).predict(Xc[te])
            print(nm, s, "제약" if use else "없음", round(time.time() - t), "s", flush=True)
        V = {"(a) 3종 모두 제약 없음": [("lgb", False), ("xgb", False), ("cat", False)],
             "(b) 3종 모두 제약": [("lgb", True), ("xgb", True), ("cat", True)],
             "(현재) LGB만 제약": [("lgb", True), ("xgb", False), ("cat", False)]}
        for k, ms in V.items():
            mean = np.mean([P[m] for m in ms], axis=0); Lm = cls(mean); t3 = cls(y)
            L = np.column_stack([cls(P[m]) for m in ms]); agree = (L == L[:, :1]).all(1); hi = agree & (Lm != 0); big = t3 != 0
            r = dv.metrics(y, mean)
            res.setdefault(k, []).append({"RMSE": r["RMSE"], "R2": r["R2"], "방향": r["방향정확도"], "3구간": np.mean(Lm == t3), "높음비율": hi.mean(),
                                         "높음정확도": np.mean(Lm[hi & big] == t3[hi & big]), **{f"{m[0]} R2": dv.metrics(y, P[m])["R2"] for m in ms}})
    base = np.array([x["RMSE"] for x in res["(현재) LGB만 제약"]])
    for k, R in res.items():
        rm = np.array([x["RMSE"] for x in R]); diff = rm - base
        row = {"분야": nm, "방식": k, **{m: round(np.mean([x[m] for x in R]), 4) for m in R[0]}, "현재 대비 ΔRMSE": f"{diff.mean():+.5f} ({(diff < 0).sum()}/5 개선)"}
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "mono_all.csv", index=False, encoding="utf-8-sig")
    del df
print(pd.DataFrame(rows).to_string(index=False))
