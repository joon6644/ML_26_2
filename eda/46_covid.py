# 18차: M17 코로나 기간 학습 가중 0.5. 최종 학습 설정 위 LGB, 시드 0 → 애매하면 1·2
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
    cov = ((df.date >= "2020-03-01") & (df.date <= "2021-12-31")).values
    w = np.where(cov, 0.5, 1.0)
    cfg = dv.TRAIN_CFG[d]
    def run(s, weight):
        X = dv.matrix(df, num, cats, tr, s)
        yt = yv[tr] if not cfg.get("clip") else np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"]))
        p = {**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s}
        if cfg.get("mono"):
            p["monotone_constraints"] = [dv.MONO[d].get(c, 0) for c in X.columns]; p["monotone_constraints_method"] = "advanced"
        m = lgb.LGBMRegressor(**p).fit(X[tr], yt, sample_weight=None if weight is None else weight[tr], categorical_feature=cats)
        return dv.metrics(y, m.predict(X[te]))
    B, M = {0: run(0, None)}, {0: run(0, w)}
    if sig <= abs(M[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
        for s in (1, 2): B[s] = run(s, None); M[s] = run(s, w)
    dd = np.mean([M[s]["RMSE"] - B[s]["RMSE"] for s in M])
    r = {"분야": nm, "시드": ",".join(map(str, M)), "학습행 중 코로나기 %": round(cov[tr].mean() * 100, 1), "R2 기준": round(np.mean([B[s]["R2"] for s in B]), 4),
         "R2 M17": round(np.mean([M[s]["R2"] for s in M]), 4), "ΔRMSE": round(dd, 5),
         "판정": "차이 없음" if abs(dd) < sig or (len(M) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")}
    rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "covid_weight.csv", index=False, encoding="utf-8-sig")
    del df
