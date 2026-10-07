# 11차: D27 aT 괴리(농), M13 느린 학습(전 분야). 기준 = 기본 + 채택 + 분야별 학습 설정, LGB
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
SLOW = {"params": {"learning_rate": 0.025}}
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.add_derived7(dv.build_all(d), d); tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    slow = {**dv.TRAIN_CFG[d], "params": {**dv.TRAIN_CFG[d].get("params", {}), "learning_rate": 0.025}}
    V = {"M13 학습률 .025·1200그루": (num, slow, 1200)}
    if d == "agri": V["+ D27 at_gap_anom"] = (num + ["at_gap_anom"], None, None)
    B = {0: dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, 0)[0])}
    for k, (n, cfg, ne) in V.items():
        run = lambda s: dv.metrics(y, dv.train_predict(df, d, n, cats, tr, te, s, cfg=cfg, n_estimators=ne)[0])
        R = {0: run(0)}
        if sig <= abs(R[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                if s not in B: B[s] = dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, s)[0])
                R[s] = run(s)
        ks = list(R); dd = np.mean([R[s]["RMSE"] - B[s]["RMSE"] for s in ks])
        row = {"분야": nm, "변형": k, "시드": ",".join(map(str, ks)), **{m: round(np.mean([R[s][m] for s in ks]), 4) for m in ["RMSE", "R2", "방향정확도"]},
               "ΔRMSE": round(dd, 5), "ΔR2": round(np.mean([R[s]["R2"] - B[s]["R2"] for s in ks]), 4),
               "판정": "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")}
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "batch11.csv", index=False, encoding="utf-8-sig")
    del df
