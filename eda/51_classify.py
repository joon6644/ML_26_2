# 4라운드: 분류 모델 묶음 C1(3진)·C2(이진)·C3(순서형) vs 회귀. 시드 42, 저장된 피처 사용
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
SEED, B = 42, 0.02
rows, times = [], []


def cls3(v): return np.where(v > B, 2, np.where(v < -B, 0, 1))


def report(nm, name, y, P3=None, pup=None, reg=None):
    """P3: (n,3) 하락·보합·상승 확률 / pup: 상승(y>0) 확률 / reg: 회귀 예측."""
    big = np.abs(y) > B; t3 = cls3(y); r = {"분야": nm, "방법": name}
    if reg is not None:
        d_ = np.sign(reg); c3 = cls3(reg)
    elif P3 is not None:
        d_ = np.where(P3[:, 2] >= P3[:, 0], 1, -1); c3 = P3.argmax(1)
    else:
        d_ = np.where(pup >= 0.5, 1, -1); c3 = None
    r["방향정확도(|y|>2%)"] = round(np.mean(d_[big] == np.sign(y[big])), 4)
    r["부호 정확도(전체)"] = round(np.mean((d_ > 0) == (y > 0)), 4)
    if c3 is not None:
        r["3구간 정확도"] = round(np.mean(c3 == t3), 4); r["3구간 macro-F1"] = round(f1_score(t3, c3, average="macro"), 4)
    if pup is not None:
        r["AUC(y>0)"] = round(roc_auc_score(y > 0, pup), 4)
    # 확신할 때만 표시: 표시 비율 / 방향정확도
    if reg is not None:
        conf, ths = np.abs(reg), [(0.02, "|ŷ|≥2%"), (0.05, "|ŷ|≥5%")]
    elif P3 is not None:
        conf, ths = np.maximum(P3[:, 2], P3[:, 0]), [(0.5, "P≥0.5"), (0.6, "P≥0.6")]
    else:
        conf, ths = np.maximum(pup, 1 - pup), [(0.6, "P≥0.6"), (0.7, "P≥0.7")]
    for t, lab in ths:
        m = conf >= t
        r[f"표시 {lab}"] = f"{m.mean():.3f} / {np.mean(d_[m & big] == np.sign(y[m & big])):.3f}" if (m & big).any() else "-"
    rows.append(r); print(r, flush=True)


def fit_lgb(X, yt, tr, te, cats, obj, **kw):
    p = {**dv.LGB_BASE, "random_state": SEED, **kw}
    if obj == "multi":
        m = lgb.LGBMClassifier(**p, objective="multiclass", num_class=3)
    elif obj == "bin":
        m = lgb.LGBMClassifier(**p, objective="binary")
    return m.fit(X[tr], yt[tr], categorical_feature=cats).predict_proba(X[te])


for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    print(nm, "피처 불러오기", round(time.time() - t0), "s", flush=True)
    X = dv.matrix(df, num, cats, tr, SEED)
    Xc = X.copy()
    for c in cats: Xc[c] = Xc[c].astype(str)
    y3, yb = cls3(yv), (yv > 0).astype(int)
    T = {}
    t = time.time(); reg = dv.train_predict(df, d, num, cats, tr, te, SEED)[0]; T["회귀 LGB"] = time.time() - t
    report(nm, "기준: 회귀 LightGBM (최종 학습 설정)", y, reg=reg)
    # C1 3진
    t = time.time(); p_l = fit_lgb(X, y3, tr, te, cats, "multi"); T["C1 LGB"] = time.time() - t; report(nm, "C1 3진 LightGBM", y, P3=p_l)
    t = time.time(); p_lb = fit_lgb(X, y3, tr, te, cats, "multi", class_weight="balanced"); T["C1 LGB 균형"] = time.time() - t; report(nm, "C1 3진 LightGBM 클래스 균형", y, P3=p_lb)
    t = time.time()
    p_x = xgb.XGBClassifier(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                            enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=SEED, objective="multi:softprob").fit(X[tr], y3[tr]).predict_proba(X[te])
    T["C1 XGB"] = time.time() - t; report(nm, "C1 3진 XGBoost", y, P3=p_x)
    t = time.time()
    p_c = CatBoostClassifier(iterations=800, learning_rate=0.1, depth=8, loss_function="MultiClass", random_seed=SEED, verbose=0, thread_count=12).fit(Xc[tr], y3[tr], cat_features=cats).predict_proba(Xc[te])
    T["C1 Cat"] = time.time() - t; report(nm, "C1 3진 CatBoost", y, P3=p_c)
    report(nm, "C1 3진 확률 평균 (LGB+XGB+Cat)", y, P3=(p_l + p_x + p_c) / 3)
    # C2 이진
    t = time.time(); b_l = fit_lgb(X, yb, tr, te, cats, "bin")[:, 1]; T["C2 LGB"] = time.time() - t; report(nm, "C2 이진 LightGBM (y>0)", y, pup=b_l)
    t = time.time()
    b_x = xgb.XGBClassifier(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                            enable_categorical=True, max_cat_to_onehot=1, n_jobs=12, random_state=SEED, objective="binary:logistic").fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
    T["C2 XGB"] = time.time() - t; report(nm, "C2 이진 XGBoost", y, pup=b_x)
    t = time.time()
    b_c = CatBoostClassifier(iterations=800, learning_rate=0.1, depth=8, loss_function="Logloss", random_seed=SEED, verbose=0, thread_count=12).fit(Xc[tr], yb[tr], cat_features=cats).predict_proba(Xc[te])[:, 1]
    T["C2 Cat"] = time.time() - t; report(nm, "C2 이진 CatBoost", y, pup=b_c)
    report(nm, "C2 이진 확률 평균", y, pup=(b_l + b_x + b_c) / 3)
    # C3 순서형
    t = time.time()
    pa = fit_lgb(X, (yv > -B).astype(int), tr, te, cats, "bin")[:, 1]; pb = fit_lgb(X, (yv > B).astype(int), tr, te, cats, "bin")[:, 1]
    T["C3 LGB 2개"] = time.time() - t
    P3o = np.column_stack([1 - pa, np.clip(pa - pb, 0, None), pb]); report(nm, "C3 순서형 LightGBM", y, P3=P3o)
    for k, v in T.items(): times.append({"분야": nm, "모델": k, "학습+예측 초": round(v, 1)})
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "classify.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(times).to_csv(ROOT / "eda" / "tables" / "classify_time.csv", index=False, encoding="utf-8-sig")
    np.savez(ROOT / "eda" / "tables" / f"classify_proba_{d}.npz", reg=reg, p_l=p_l, p_lb=p_lb, p_x=p_x, p_c=p_c, b_l=b_l, b_x=b_x, b_c=b_c, P3o=P3o)
    del df, X, Xc
