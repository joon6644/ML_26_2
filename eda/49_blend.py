# 2라운드 B·C + 추론 시간: 가중치·메타 모델은 2016~2021 학습 → 2022 예측으로만 정함(검증셋 미사용)
# 인자: 분야들, 채택된 A 그룹(JSON: {"agri": ["G1"], ...})
import importlib.util, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import nnls
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
src = open(ROOT / "eda" / "32_ensemble.py", encoding="utf-8").read().split("def nat_oof")[0]
exec(src)   # clip_y, xgb_pred, cat_pred
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
TYPED = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
rows, times = [], []


def level1(df, d, num, cats, fit_m, pred_m):
    """LGB 5시드 평균(분야 설정), XGB, Cat을 fit_m으로 학습해 pred_m 예측. 학습·예측 시간 기록."""
    out, tm = {}, {}
    t = time.time(); ps = []; pt = 0
    for s in range(5):
        p, m, X = dv.train_predict(df, d, num, cats, fit_m, pred_m, s)
        t1 = time.time(); m.predict(X[pred_m]); pt += time.time() - t1
        ps.append(p)
    out["lgb5"] = np.mean(ps, axis=0); tm["lgb5"] = (time.time() - t, pt)
    X = dv.matrix(df, num, cats, fit_m, 0); yc = clip_y(d, df.y.values, fit_m)
    huber = dv.TRAIN_CFG[d].get("params", {}).get("objective") == "huber"
    t = time.time()
    mx = xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                          enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=0,
                          **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {})).fit(X[fit_m], yc[fit_m])
    t1 = time.time(); out["xgb"] = mx.predict(X[pred_m]); tm["xgb"] = (time.time() - t, time.time() - t1)
    Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].astype(str)
    t = time.time()
    mc = CatBoostRegressor(iterations=800, learning_rate=0.1, depth=8, loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=0, verbose=0, thread_count=12).fit(Xc[fit_m], yc[fit_m], cat_features=cats)
    t1 = time.time(); out["cat"] = mc.predict(Xc[pred_m]); tm["cat"] = (time.time() - t, time.time() - t1)
    return out, tm


for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.build_all(d)
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]
    if TYPED.get(d):
        df, G = dv.add_typed(df, d)
        for g in TYPED[d]:
            if g == "G1′":
                num = [c for c in num if c not in dv.typed_groups(d, nm)["G1"]] + G["G1"]
            else:
                num = num + G[g]
    tr, te = dv.masks(df); yr = df.date.dt.year.values; yv = df.y.values
    tr_in, va = tr & (yr <= 2021), tr & (yr == 2022)
    P22, _ = level1(df, d, num, cats, tr_in, va)
    Pte, tm = level1(df, d, num, cats, tr, te)
    names = ["lgb5", "xgb", "cat"]
    A22 = np.column_stack([P22[n] for n in names]); Ate = np.column_stack([Pte[n] for n in names])
    w, _ = nnls(A22, yv[va]); 
    y = yv[te]
    res = {"LGB5 단독": Pte["lgb5"], "단순 평균 (LGB5+XGB+Cat)": Ate.mean(1), f"B 가중 평균 w={np.round(w, 2).tolist()}": Ate @ w}
    meta_X22 = pd.DataFrame(A22, columns=names).assign(fg=df.food_group.values[va], nat=df.is_nat.values[va])
    meta_Xte = pd.DataFrame(Ate, columns=names).assign(fg=df.food_group.values[te], nat=df.is_nat.values[te])
    for X_ in (meta_X22, meta_Xte): X_["fg"] = X_.fg.astype(pd.CategoricalDtype(sorted(df.food_group.astype(str).unique())))
    meta = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=7, min_child_samples=500, random_state=0, verbose=-1).fit(meta_X22, yv[va], categorical_feature=["fg"])
    res["C 스태킹 (깊이3 메타, 계열·전국 여부)"] = meta.predict(meta_Xte)
    for k, p in res.items():
        r = {"분야": nm, "방법": k, **{a: round(b, 4) for a, b in dv.metrics(y, p).items()}}
        rows.append(r); print(r, flush=True)
    ndays = df.loc[te, "date"].nunique(); n_te = te.sum()
    for k, (fit_t, pred_t) in tm.items():
        times.append({"분야": nm, "모델": k, "학습 초(전체 학습기간)": round(fit_t, 1), "검증 전체 예측 초": round(pred_t, 2) if pred_t == pred_t else None,
                      "검증 행": int(n_te), "하루 평균 행": round(n_te / ndays)})
    # 하루치 예측 시간 실측 (LGB 1개·XGB·Cat 모두 새로 학습하지 않고, 하루 행만 예측)
    day = df.loc[te, "date"].max(); dm = te & (df.date == day).values
    X = dv.matrix(df, num, cats, tr, 0); yc = clip_y(d, yv, tr)
    m = lgb.LGBMRegressor(**{**dv.LGB_BASE, **dv.TRAIN_CFG[d].get("params", {})}).fit(X[tr], yc[tr], categorical_feature=cats)
    t = time.time(); m.predict(X[dm]); t_l = time.time() - t
    times.append({"분야": nm, "모델": "하루치 예측 실측 (LGB 1개)", "검증 전체 예측 초": round(t_l, 3), "검증 행": int(dm.sum()), "하루 평균 행": int(dm.sum())})
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "blend.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(times).to_csv(ROOT / "eda" / "tables" / "inference_time.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([t_ for t_ in times if t_["분야"] == nm]).to_string(index=False), flush=True)
    del df, X
