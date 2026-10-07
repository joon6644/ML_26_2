# 3라운드: 다양성 앙상블 (서비스 기준 단일 시드 42 — 다중 시드는 확인용으로만 사용)
# (부스팅 3종 + Ridge + ExtraTrees + RandomForest), 다년 전진 교차 예측으로 조합 추정
import importlib.util, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from scipy import sparse
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
NAMES = ["lgb", "xgb", "cat", "ridge", "et", "rf"]
SEED = 42
rows, times = [], []


def ytrain(d, y, m):
    c = dv.TRAIN_CFG[d].get("clip") or ((0.01, 0.99) if dv.TRAIN_CFG[d].get("params", {}).get("objective") == "huber" else None)
    return np.clip(y[m], *np.nanquantile(y[m], c)) if c else y[m]   # Huber 쓰는 분야의 비부스팅 모델은 1~99% clip으로 대신


def fit_all(df, d, num, cats, fit_m, pred_m, models=NAMES):
    out, tm = {}, {}
    yv = df.y.values
    huber = dv.TRAIN_CFG[d].get("params", {}).get("objective") == "huber"
    if "lgb" in models:
        t = time.time()
        out["lgb"] = dv.train_predict(df, d, num, cats, fit_m, pred_m, SEED)[0]; tm["lgb"] = time.time() - t
    X = dv.matrix(df, num, cats, fit_m, SEED)
    ycl = dv.TRAIN_CFG[d].get("clip")
    yb = np.clip(yv[fit_m], *np.nanquantile(yv[fit_m], ycl)) if ycl else yv[fit_m]
    if "xgb" in models:
        t = time.time()
        out["xgb"] = xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                                      enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=SEED,
                                      **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {})).fit(X[fit_m], yb).predict(X[pred_m]); tm["xgb"] = time.time() - t
    if "cat" in models:
        t = time.time(); Xc = X.copy()
        for c in cats: Xc[c] = Xc[c].astype(str)
        out["cat"] = CatBoostRegressor(iterations=800, learning_rate=0.1, depth=8, loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=SEED, verbose=0,
                                       thread_count=12).fit(Xc[fit_m], yb, cat_features=cats).predict(Xc[pred_m]); tm["cat"] = time.time() - t
    numX = X[[c for c in X.columns if c not in cats]].values.astype("float32")
    yo = ytrain(d, yv, fit_m)
    if "ridge" in models:
        t = time.time()
        oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(df.loc[fit_m, cats].astype(str))
        M = lambda m: sparse.hstack([sparse.csr_matrix(numX[m]), oh.transform(df.loc[m, cats].astype(str))]).tocsr()
        out["ridge"] = Ridge(alpha=1.0).fit(M(fit_m), yo).predict(M(pred_m)); tm["ridge"] = time.time() - t
    codes = np.column_stack([df[c].astype("category").cat.codes.values for c in cats]).astype("float32")
    Z = np.hstack([numX, codes])
    for name, cls, kw in [("et", ExtraTreesRegressor, {"bootstrap": True}), ("rf", RandomForestRegressor, {})]:
        if name in models:
            t = time.time()
            m = cls(n_estimators=100, min_samples_leaf=200, max_features=0.5, max_samples=0.25, n_jobs=12, random_state=SEED, **({"bootstrap": True} if name == "rf" else kw))
            out[name] = m.fit(Z[fit_m], yo).predict(Z[pred_m]); tm[name] = time.time() - t
    return out, tm


for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df, num, cats = dv.final_frame(d); tr, te = dv.masks(df); yv = df.y.values; y = yv[te]; yr = df.date.dt.year.values
    Pte, tm = fit_all(df, d, num, cats, tr, te)
    for k, v in tm.items(): times.append({"분야": nm, "모델": k, "학습+예측 초": round(v, 1)})
    base_name = "LGB+XGB+Cat (현재 농산물 최종)" if d == "agri" else "LGB (현재 최종)"
    base_p = (Pte["lgb"] + Pte["xgb"] + Pte["cat"]) / 3 if d == "agri" else Pte["lgb"]
    err = {k: v - y for k, v in Pte.items()}
    for k, p in Pte.items():
        rows.append({"분야": nm, "방법": f"단독: {k}", **{a: round(b, 4) for a, b in dv.metrics(y, p).items()}, "LGB와 오차 상관": round(np.corrcoef(err[k], err["lgb"])[0, 1], 3)})
        print(rows[-1], flush=True)
    combos = {"부스팅 3종 평균": ["lgb", "xgb", "cat"], "+ ridge": ["lgb", "xgb", "cat", "ridge"], "+ et": ["lgb", "xgb", "cat", "et"], "+ rf": ["lgb", "xgb", "cat", "rf"],
              "+ ridge·et·rf (6종)": NAMES, "LGB + et": ["lgb", "et"], "LGB + ridge": ["lgb", "ridge"], "LGB + et + ridge": ["lgb", "et", "ridge"]}
    rows.append({"분야": nm, "방법": f"기준: {base_name}", **{a: round(b, 4) for a, b in dv.metrics(y, base_p).items()}}); print(rows[-1], flush=True)
    for k, ms in combos.items():
        rows.append({"분야": nm, "방법": f"E2 균등: {k}", **{a: round(b, 4) for a, b in dv.metrics(y, np.mean([Pte[m] for m in ms], axis=0)).items()}}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "diverse.csv", index=False, encoding="utf-8-sig")
    # E3: 다년 전진 교차 예측 (2020·2021·2022)
    t = time.time()
    oof_P, oof_y, oof_fg, oof_nat = [], [], [], []
    for Y in (2020, 2021, 2022):
        fm, pm = tr & (yr < Y), tr & (yr == Y)
        if fm.sum() < 1000 or pm.sum() < 1000: continue
        P, _ = fit_all(df, d, num, cats, fm, pm)
        oof_P.append(np.column_stack([P[n] for n in NAMES])); oof_y.append(yv[pm]); oof_fg.append(df.food_group.values[pm]); oof_nat.append(df.is_nat.values[pm])
    A = np.vstack(oof_P); yo = np.concatenate(oof_y); Ate = np.column_stack([Pte[n] for n in NAMES])
    times.append({"분야": nm, "모델": "E3 다년 교차 예측 (3개 연도 × 6종)", "학습+예측 초": round(time.time() - t, 1)})
    w, _ = nnls(A, yo)
    rows.append({"분야": nm, "방법": f"E3 NNLS 가중 (다년) w={dict(zip(NAMES, np.round(w, 2)))}", **{a: round(b, 4) for a, b in dv.metrics(y, Ate @ w).items()}})
    rm = Ridge(alpha=10.0).fit(A, yo)
    rows.append({"분야": nm, "방법": f"E3 Ridge 메타 (다년) coef={dict(zip(NAMES, np.round(rm.coef_, 2)))}", **{a: round(b, 4) for a, b in dv.metrics(y, rm.predict(Ate)).items()}})
    fgc = pd.CategoricalDtype(sorted(df.food_group.astype(str).unique()))
    MX = pd.DataFrame(A, columns=NAMES).assign(fg=pd.Series(np.concatenate(oof_fg)).astype(str).astype(fgc), nat=np.concatenate(oof_nat))
    MT = pd.DataFrame(Ate, columns=NAMES).assign(fg=pd.Series(df.food_group.values[te]).astype(str).astype(fgc), nat=df.is_nat.values[te])
    meta = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, num_leaves=4, min_child_samples=1000, random_state=SEED, verbose=-1).fit(MX, yo, categorical_feature=["fg"])
    rows.append({"분야": nm, "방법": "E3 LightGBM 메타 깊이2 (다년, 계열·전국 여부)", **{a: round(b, 4) for a, b in dv.metrics(y, meta.predict(MT)).items()}})
    for r in rows[-3:]: print(r, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "diverse.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(times).to_csv(ROOT / "eda" / "tables" / "diverse_time.csv", index=False, encoding="utf-8-sig")
    print(nm, "총", round(time.time() - t0), "s", pd.DataFrame([x for x in times if x["분야"] == nm]).to_string(index=False), flush=True)
    del df
