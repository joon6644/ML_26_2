# LGB+XGB+Cat vs LGB+XGB+RF. 인자: 단조 제약을 3종 모두에 적용할지 (1/0). 농산물은 저장된 시드 42·0·1 예측 재사용
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
MONO_ALL = sys.argv[1] == "1" if len(sys.argv) > 1 else False
cls = lambda v: np.where(v > .02, 1, np.where(v < -.02, -1, 0))
rows, times = [], []


def rf_pred(df, d, num, cats, X, tr, te, yt, s, mono):
    numc = [c for c in X.columns if c not in cats]
    Z = np.hstack([X[numc].values.astype("float32"), np.column_stack([df[c].astype("category").cat.codes.values for c in cats]).astype("float32")])
    cst = [mono.get(c, 0) for c in numc] + [0] * len(cats) if mono else None
    m = RandomForestRegressor(n_estimators=100, min_samples_leaf=200, max_features=0.5, max_samples=0.25, bootstrap=True, n_jobs=12, random_state=s,
                              **({"monotonic_cst": cst} if cst else {}))
    return m.fit(Z[tr], yt).predict(Z[te])


for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]; huber = cfg.get("params", {}).get("objective") == "huber"
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    use_mono = (d != "agri") and (MONO_ALL or cfg.get("mono"))
    mono = {k: v for k, v in dv.MONO[d].items() if k in num} if use_mono else {}
    seeds = [42, 0, 1] if d == "agri" else [42, 0, 1, 2, 3]
    res = {}
    for s in seeds:
        X = dv.matrix(df, num, cats, tr, s)
        if d == "agri":
            P = dict(np.load(ROOT / "eda" / "tables" / f"preds_{d}_s{s}.npz"))
        else:
            Xc = X.copy()
            for c in cats: Xc[c] = Xc[c].astype(str)
            cons = [mono.get(c, 0) for c in X.columns]
            pl = {**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s}
            if mono: pl.update(monotone_constraints=cons, monotone_constraints_method="advanced")
            P = {"lgb": lgb.LGBMRegressor(**pl).fit(X[tr], yt, categorical_feature=cats).predict(X[te])}
            px = dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                      enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=s, **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {}))
            if mono and MONO_ALL: px["monotone_constraints"] = dict(mono)
            P["xgb"] = xgb.XGBRegressor(**px).fit(X[tr], yt).predict(X[te])
            t = time.time()
            pc = dict(iterations=800, learning_rate=0.1, depth=8, loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=s, verbose=0, thread_count=12)
            if mono and MONO_ALL: pc["monotone_constraints"] = dict(mono)
            P["cat"] = CatBoostRegressor(**pc).fit(Xc[tr], yt, cat_features=cats).predict(Xc[te]); times.append({"분야": nm, "모델": "cat", "초": time.time() - t})
        t = time.time(); P["rf"] = rf_pred(df, d, num, cats, X, tr, te, yt, s, mono if MONO_ALL else {}); times.append({"분야": nm, "모델": "rf", "초": time.time() - t})
        for k, ms in [("LGB+XGB+Cat", ["lgb", "xgb", "cat"]), ("LGB+XGB+RF", ["lgb", "xgb", "rf"])]:
            mean = np.mean([P[m] for m in ms], axis=0); Lm = cls(mean); t3 = cls(y)
            L = np.column_stack([cls(P[m]) for m in ms]); agree = (L == L[:, :1]).all(1); hi = agree & (Lm != 0); big = t3 != 0
            r = dv.metrics(y, mean)
            res.setdefault(k, []).append({"RMSE": r["RMSE"], "R2": r["R2"], "방향": r["방향정확도"], "3구간": np.mean(Lm == t3), "만장일치": agree.mean(),
                                         "높음비율": hi.mean(), "높음정확도": np.mean(Lm[hi & big] == t3[hi & big]), "3번째 모델 단독 R2": dv.metrics(y, P[ms[2]])["R2"]})
        print(nm, s, {k: round(v[-1]["RMSE"], 5) for k, v in res.items()}, flush=True)
    b = np.array([x["RMSE"] for x in res["LGB+XGB+Cat"]])
    for k, R in res.items():
        diff = np.array([x["RMSE"] for x in R]) - b
        rows.append({"분야": nm, "구성": k, "시드 수": len(R), **{m: round(np.mean([x[m] for x in R]), 4) for m in R[0]}, "Cat 대비 ΔRMSE": f"{diff.mean():+.5f} ({(diff < 0).sum()}/{len(diff)} 개선)"})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "rf_vs_cat.csv", index=False, encoding="utf-8-sig")
    del df
T = pd.DataFrame(times).groupby(["분야", "모델"]).초.mean().round(1); print(T)
