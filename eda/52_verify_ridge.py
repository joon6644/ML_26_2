# 확인용: 축·수 '부스팅 3종 + Ridge' vs LGB 단독 vs 부스팅 3종 — 시드 0, 1 추가 (서비스는 42)
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "50_diverse.py", encoding="utf-8").read().split("for d in DOMS:")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d in ["livestock", "fishery"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); y = df.y.values[te].astype(float)
    for s in (42, 0, 1):
        SEED = s
        P, _ = fit_all(df, d, num, cats, tr, te, models=["lgb", "xgb", "cat", "ridge"])
        for lab, ms in [("LGB 단독", ["lgb"]), ("부스팅 3종", ["lgb", "xgb", "cat"]), ("부스팅 3종 + Ridge", ["lgb", "xgb", "cat", "ridge"]), ("LGB + Ridge", ["lgb", "ridge"])]:
            r = dv.metrics(y, np.mean([P[m] for m in ms], axis=0)); rows.append({"분야": nm, "시드": s, "방식": lab, "RMSE": r["RMSE"], "R2": r["R2"], "방향정확도": r["방향정확도"]})
        print(nm, s, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(ROOT / "eda" / "tables" / "verify_ridge.csv", index=False, encoding="utf-8-sig")
print(R.pivot_table(index=["분야", "방식"], columns="시드", values="RMSE").round(5).to_string())
print(R.groupby(["분야", "방식"])[["RMSE", "R2", "방향정확도"]].mean().round(4).to_string())
