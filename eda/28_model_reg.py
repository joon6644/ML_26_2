# 3차: 모델 설정 가설 M2(강한 규제) · M3(최근 연도 가중). 피처 = 기본 + 채택 누적(dv.ADOPTED). 시드 0
import importlib.util, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
spec = importlib.util.spec_from_file_location("mc", ROOT / "eda" / "26_models.py")
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(dv.cs.rm.NAMES)
BASEP = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
CFG = {
    "기본": ({}, None),
    "R1 잎31·최소500": ({"num_leaves": 31, "min_child_samples": 500}, None),
    "R2 잎15·최소2000·800그루": ({"num_leaves": 15, "min_child_samples": 2000, "n_estimators": 800}, None),
    "R3 잎63·최소1000·열50%·L2 10": ({"min_child_samples": 1000, "colsample_bytree": 0.5, "reg_lambda": 10}, None),
    "R4 잎31·최소200·300그루": ({"num_leaves": 31, "min_child_samples": 200, "n_estimators": 300}, None),
    "W1 연도 선형 가중": ({}, "linear"),
    "W2 반감기 2년 가중": ({}, "half2"),
}


def matrix(df, num, cats, tr):
    num = sorted(num)
    X = dv.cs.rm.prep(df, num, tr)
    for c, ok in df.attrs.get("scope", {}).items():
        if c in X: X.loc[~ok, c] = -1.0
    for c in cats:
        X[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    return X


rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.cs.build(d)[0]
    df = dv.add_derived2(dv.add_derived(df, d), d)
    tr, te = dv.masks(df); y = df.y.values
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]
    X = matrix(df, num, cats, tr)
    yr = df.date.dt.year.values[tr]
    for name, (p, wmode) in CFG.items():
        t = time.time()
        w = None if wmode is None else ((yr - 2015).astype(float) if wmode == "linear" else 0.5 ** ((2022 - yr) / 2.0))
        m = lgb.LGBMRegressor(**{**BASEP, **p}).fit(X[tr], y[tr], sample_weight=w, categorical_feature=cats)
        r = {"분야": nm, "설정": name, **{k: round(v, 4) for k, v in dv.metrics(y[te], m.predict(X[te])).items()}, "초": round(time.time() - t)}
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "model_reg.csv", index=False, encoding="utf-8-sig")
    del df, X
