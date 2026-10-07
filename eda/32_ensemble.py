# 6차: M8 알고리즘 앙상블, M9 시드 앙상블, M10 2단계(전국 모델 OOF 예측을 피처로). 기준 = 기본 + 채택 + 분야별 학습 설정
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
rows = []


def clip_y(d, y, tr):
    c = dv.TRAIN_CFG[d].get("clip")
    return np.clip(y, *np.nanquantile(y[tr], c)) if c else y


def xgb_pred(d, X, y, tr, te, cats):
    huber = dv.TRAIN_CFG[d].get("params", {}).get("objective") == "huber"
    m = xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                         enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=0,
                         **({"objective": "reg:pseudohubererror", "huber_slope": 0.05} if huber else {}))
    return m.fit(X[tr], y[tr]).predict(X[te])


def cat_pred(d, X, y, tr, te, cats):
    huber = dv.TRAIN_CFG[d].get("params", {}).get("objective") == "huber"
    Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].astype(str)
    m = CatBoostRegressor(iterations=800, learning_rate=0.1, depth=8, loss_function="Huber:delta=0.05" if huber else "RMSE", random_seed=0, verbose=0, thread_count=12)
    return m.fit(Xc[tr], y[tr], cat_features=cats).predict(Xc[te])


def nat_oof(df, d, num, cats, tr):
    """전국 행만으로 학습한 모델의 예측: 학습기간은 연도별 교차(그 해 제외 학습), 검증기간은 학습기간 전체로 학습."""
    nat = df.is_nat.values == 1
    yr = df.date.dt.year.values
    ncats = [c for c in cats if c != "sgg_nm"]
    nnum = [c for c in num if not c.startswith("nat_") and c != "premium"]
    X = dv.matrix(df, nnum, ncats, tr)
    y = clip_y(d, df.y.values, tr)
    pred = np.full(len(df), np.nan)
    p = {**dv.LGB_BASE, **dv.TRAIN_CFG[d].get("params", {})}
    years = sorted(set(yr[tr & nat]))
    for Y in years:
        fit_m = tr & nat & (yr != Y)
        m = lgb.LGBMRegressor(**p).fit(X[fit_m], y[fit_m], categorical_feature=ncats)
        pred[tr & nat & (yr == Y)] = m.predict(X[tr & nat & (yr == Y)])
    m = lgb.LGBMRegressor(**p).fit(X[tr & nat], y[tr & nat], categorical_feature=ncats)
    te_nat = ~tr & nat
    pred[te_nat] = m.predict(X[te_nat])
    s = pd.DataFrame({k: df[k].values for k in dv.cs.KEYS + ["date"]}).assign(p=pred)[nat]
    return df[dv.cs.KEYS + ["date"]].merge(s, on=dv.cs.KEYS + ["date"], how="left").p.values   # 지역 행에도 같은 시계열·날짜의 전국 예측


for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df = dv.add_derived(dv.cs.build(d)[0], d)
    tr, te = dv.masks(df); yv = df.y.values; y = yv[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]
    res = {}
    P = {}
    for s in range(5):
        P[f"lgb{s}"] = dv.train_predict(df, d, num, cats, tr, te, s)[0]
    res["기준 LightGBM (시드 0)"] = P["lgb0"]
    res["M9 시드 앙상블 (LightGBM 5개)"] = np.mean([P[f"lgb{s}"] for s in range(5)], axis=0)
    X = dv.matrix(df, num, cats, tr, 0); yc = clip_y(d, yv, tr)
    t = time.time(); P["xgb"] = xgb_pred(d, X, yc, tr, te, cats); print(nm, "xgb", round(time.time() - t), "s", flush=True)
    t = time.time(); P["cat"] = cat_pred(d, X, yc, tr, te, cats); print(nm, "cat", round(time.time() - t), "s", flush=True)
    res["XGBoost 단독"] = P["xgb"]; res["CatBoost 단독"] = P["cat"]
    res["M8 알고리즘 앙상블 (LGB+XGB+Cat)"] = (P["lgb0"] + P["xgb"] + P["cat"]) / 3
    res["M8+M9 (LGB 5시드 평균 + XGB + Cat)"] = (res["M9 시드 앙상블 (LightGBM 5개)"] + P["xgb"] + P["cat"]) / 3
    t = time.time()
    df["nat_model_pred"] = nat_oof(df, d, num, cats, tr)
    print(nm, "M10 oof", round(time.time() - t), "s | 결측%", round(df.nat_model_pred.isna().mean() * 100, 1), flush=True)
    for s in range(3):
        P[f"m10_{s}"] = dv.train_predict(df, d, num + ["nat_model_pred"], cats, tr, te, s)[0]
    res["M10 2단계 (시드 0)"] = P["m10_0"]
    res["M10 2단계 시드 0~2 vs 기준 시드 0~2"] = np.mean([P[f"m10_{s}"] for s in range(3)], axis=0)
    base3 = np.mean([P[f"lgb{s}"] for s in range(3)], axis=0)
    for k, p in res.items():
        r = {"분야": nm, "모델": k, **{a: round(b, 4) for a, b in dv.metrics(y, p).items()}}
        rows.append(r); print(r, flush=True)
    m10s = [dv.metrics(y, P[f"m10_{s}"])["RMSE"] - dv.metrics(y, P[f"lgb{s}"])["RMSE"] for s in range(3)]
    rows.append({"분야": nm, "모델": "참고: M10 − 기준 RMSE (시드별)", "RMSE": " / ".join(f"{x:+.5f}" for x in m10s)})
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "ensemble.csv", index=False, encoding="utf-8-sig")
    del df, X
