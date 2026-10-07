# 9차: M12 전국 행 학습 가중. 기준 = 기본 + 채택 + 분야별 학습 설정, LGB 단일
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df = dv.build_all(d); tr, te = dv.masks(df); yv = df.y.values; y = yv[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    nreg = df[df.is_nat == 0].groupby(dv.cs.KEYS + ["date"]).size().rename("nreg").reset_index()
    w = df[dv.cs.KEYS + ["date", "is_nat"]].merge(nreg, on=dv.cs.KEYS + ["date"], how="left")
    w = np.where(w.is_nat == 1, w.nreg.fillna(1).clip(lower=1), 1.0)
    cfg = dv.TRAIN_CFG[d]
    def run(s, weight):
        X = dv.matrix(df, num, cats, tr, s)
        yt = yv[tr] if not cfg.get("clip") else np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"]))
        m = lgb.LGBMRegressor(**{**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s}).fit(X[tr], yt, sample_weight=weight[tr] if weight is not None else None, categorical_feature=cats)
        p = m.predict(X[te]); nat = df.is_nat.values[te] == 1
        r = dv.metrics(y, p); r["R2 전국"] = dv.metrics(y[nat], p[nat])["R2"]; r["R2 지역"] = dv.metrics(y[~nat], p[~nat])["R2"]
        return r
    B, M = {0: run(0, None)}, {0: run(0, w)}
    if sig <= abs(M[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
        for s in (1, 2): B[s] = run(s, None); M[s] = run(s, w)
    for lab, R in [("기준", B), ("M12 전국 행 가중", M)]:
        rr = {"분야": nm, "변형": lab, "시드": ",".join(map(str, R)), **{k: round(np.mean([R[s][k] for s in R]), 4) for k in R[0]}}
        rows.append(rr); print(rr, flush=True)
    print("ΔRMSE", round(np.mean([M[s]["RMSE"] - B[s]["RMSE"] for s in M]), 5), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "m12.csv", index=False, encoding="utf-8-sig")
    del df
