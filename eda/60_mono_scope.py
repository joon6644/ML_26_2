# 단조 제약 범위: 농 = 없음 vs 상대값 9개 / 축·수 = 현재(일치 전부) vs 수준값 제외. LGB·XGB·Cat 모두 같은 제약, 균등 평균 + 만장일치 신뢰도
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
cls = lambda v: np.where(v > .02, 1, np.where(v < -.02, -1, 0))
LEVEL = {"price_kg", "garak_price_kg", "whsl_price_kg", "mafra_whsl_price_kg", "census_heads", "auction_heads", "auction_price_kg", "catch_ton",
         "catch_value_kkrw", "export_kg", "import_kg", "garak_n_trades", "stock_ton", "cpi_item"}
AGRI9 = {"dev_last": 1, "yoy": -1, "r91": -1, "premium": -1, "nat_dev_last": 1, "nat_yoy": -1, "nat_r91": -1, "whsl_gap_anom": 1, "grp_r28": 1}
rows = []


def fit3(df, d, num, cats, tr, te, yt, s, mono):
    cfg = dv.TRAIN_CFG[d]; huber = cfg.get("params", {}).get("objective") == "huber"
    X = dv.matrix(df, num, cats, tr, s); Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].astype(str)
    pl = {**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s}
    if mono: pl.update(monotone_constraints=[mono.get(c, 0) for c in X.columns], monotone_constraints_method="advanced")
    P = {"lgb": lgb.LGBMRegressor(**pl).fit(X[tr], yt, categorical_feature=cats).predict(X[te])}
    px = dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
              enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=s, **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {}))
    if mono: px["monotone_constraints"] = dict(mono)
    P["xgb"] = xgb.XGBRegressor(**px).fit(X[tr], yt).predict(X[te])
    pc = dict(iterations=800, learning_rate=0.1, depth=8, loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=s, verbose=0, thread_count=12)
    if mono: pc["monotone_constraints"] = dict(mono)
    P["cat"] = CatBoostRegressor(**pc).fit(Xc[tr], yt, cat_features=cats).predict(Xc[te])
    return P


for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    if d == "agri":
        V = {"제약 없음 (현재)": None, "상대값·되돌림 9개": {k: v for k, v in AGRI9.items() if k in num}}
        seeds = [42, 0, 1]
    else:
        allm = {k: v for k, v in dv.MONO[d].items() if k in num}
        V = {"일치 피처 전부 (현재)": allm, "수준값 제외": {k: v for k, v in allm.items() if k not in LEVEL}}
        seeds = [42, 0, 1, 2, 3]
    for k, m in V.items(): print(nm, k, len(m or {}), sorted((m or {}).keys()), flush=True)
    res = {}
    for s in seeds:
        for k, m in V.items():
            t = time.time()
            if d == "agri" and m is None:
                P = dict(np.load(ROOT / "eda" / "tables" / f"preds_agri_s{s}.npz")); P = {x: P[x] for x in ["lgb", "xgb", "cat"]}
            else:
                P = fit3(df, d, num, cats, tr, te, yt, s, m)
            mean = np.mean(list(P.values()), axis=0); Lm = cls(mean); t3 = cls(y)
            L = np.column_stack([cls(v) for v in P.values()]); agree = (L == L[:, :1]).all(1); hi = agree & (Lm != 0); big = t3 != 0
            r = dv.metrics(y, mean)
            res.setdefault(k, []).append({"RMSE": r["RMSE"], "R2": r["R2"], "방향": r["방향정확도"], "3구간": np.mean(Lm == t3), "높음비율": hi.mean(),
                                         "높음정확도": np.mean(Lm[hi & big] == t3[hi & big]), **{f"{x} R2": dv.metrics(y, P[x])["R2"] for x in P}})
            print(nm, s, k, round(r["RMSE"], 5), round(time.time() - t), "s", flush=True)
    keys = list(V); b = np.array([x["RMSE"] for x in res[keys[0]]])
    for k in keys:
        R = res[k]; diff = np.array([x["RMSE"] for x in R]) - b
        rows.append({"분야": nm, "제약": k, "제약 수": len(V[k] or {}), "시드 수": len(R), **{c: round(np.mean([x[c] for x in R]), 4) for c in R[0]},
                     "현재 대비 ΔRMSE": f"{diff.mean():+.5f} ({(diff < 0).sum()}/{len(diff)} 개선)"})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "mono_scope.csv", index=False, encoding="utf-8-sig")
    del df
print(pd.DataFrame(rows).to_string(index=False))
