# 7라운드 탐색: 모델별 하이퍼파라미터 (시드 42 단일). 분야 학습 설정·단조 제약(확정 범위) 적용
import importlib.util, sys, time, itertools, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)   # LEVEL
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
rng = np.random.default_rng(42)
GRID = {
    "lgb": {"num_leaves": [31, 63, 127], "min_child_samples": [20, 100], "lr_n": [(0.05, 600), (0.03, 1000)], "colsample_bytree": [0.8, 0.6]},
    "xgb": {"max_depth": [6, 7, 8], "min_child_weight": [20, 100], "lr_n": [(0.05, 600), (0.03, 1000)], "colsample_bytree": [0.8, 0.6]},
    "cat": {"depth": [6, 8], "lr_n": [(0.1, 800), (0.05, 1500)], "l2_leaf_reg": [3, 10]},
}
DEFAULT = {"lgb": {"num_leaves": 63, "min_child_samples": 20, "lr_n": (0.05, 600), "colsample_bytree": 0.8},
           "xgb": {"max_depth": 7, "min_child_weight": 20, "lr_n": (0.05, 600), "colsample_bytree": 0.8},
           "cat": {"depth": 8, "lr_n": (0.1, 800), "l2_leaf_reg": 3}}


def combos(model, k):
    allc = [dict(zip(GRID[model], v)) for v in itertools.product(*GRID[model].values())]
    allc = [c for c in allc if c != DEFAULT[model]]
    pick = [allc[i] for i in rng.choice(len(allc), min(k, len(allc)), replace=False)]
    return [DEFAULT[model]] + pick


def fit_one(model, hp, df, d, num, cats, tr, te, yt, s, mono):
    cfg = dv.TRAIN_CFG[d]; huber = cfg.get("params", {}).get("objective") == "huber"
    X = dv.matrix(df, num, cats, tr, s)
    lr, n = hp["lr_n"]
    if model == "lgb":
        p = {**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s, "num_leaves": hp["num_leaves"], "min_child_samples": hp["min_child_samples"],
             "learning_rate": lr, "n_estimators": n, "colsample_bytree": hp["colsample_bytree"]}
        if mono: p.update(monotone_constraints=[mono.get(c, 0) for c in X.columns], monotone_constraints_method="advanced")
        return lgb.LGBMRegressor(**p).fit(X[tr], yt, categorical_feature=cats).predict(X[te])
    if model == "xgb":
        p = dict(n_estimators=n, max_depth=hp["max_depth"], learning_rate=lr, subsample=0.8, colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"],
                 tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=s, **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {}))
        if mono: p["monotone_constraints"] = dict(mono)
        return xgb.XGBRegressor(**p).fit(X[tr], yt).predict(X[te])
    Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].astype(str)
    p = dict(iterations=n, learning_rate=lr, depth=hp["depth"], l2_leaf_reg=hp["l2_leaf_reg"], loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=s, verbose=0, thread_count=12)
    if mono: p["monotone_constraints"] = dict(mono)
    return CatBoostRegressor(**p).fit(Xc[tr], yt, cat_features=cats).predict(Xc[te])


if __name__ == "__main__":
    rows = []
    for d in DOMS:
        nm = dv.cs.rm.NAMES[d]
        df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
        cfg = dv.TRAIN_CFG[d]
        yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
        mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
        for model in ["lgb", "xgb", "cat"]:
            k = 4 if (d == "agri" and model == "cat") else 8
            for hp in combos(model, k):
                t = time.time(); p = fit_one(model, hp, df, d, num, cats, tr, te, yt, 42, mono); r = dv.metrics(y, p)
                rows.append({"분야": nm, "모델": model, "설정": json.dumps(hp), "기본값": hp == DEFAULT[model], "RMSE": round(r["RMSE"], 5), "R2": round(r["R2"], 4),
                             "방향정확도": round(r["방향정확도"], 4), "초": round(time.time() - t)})
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "tune_search.csv", index=False, encoding="utf-8-sig")
        del df
