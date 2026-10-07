# 17차: 축·수 LightGBM 하이퍼파라미터 점검 (최종 학습 설정 위에서)
import importlib.util, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.build_all(d); tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]
    res = []
    for leaves, mcs, (lr, ne) in itertools.product([31, 63, 127], [20, 100], [(0.05, 600), (0.025, 1200)]):
        cfg = {**dv.TRAIN_CFG[d], "params": {**dv.TRAIN_CFG[d].get("params", {}), "num_leaves": leaves, "min_child_samples": mcs, "learning_rate": lr}}
        r = dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, 0, cfg=cfg, n_estimators=ne)[0])
        res.append(((leaves, mcs, lr, ne), r)); rows.append({"분야": nm, "잎": leaves, "최소표본": mcs, "lr": lr, "나무": ne, "시드": "0", **{k: round(v, 4) for k, v in r.items()}})
        print(rows[-1], flush=True)
    base = [x for x in res if x[0] == (63, 20, 0.05, 600)][0][1]["RMSE"]
    top = sorted(res, key=lambda x: x[1]["RMSE"])[:2]
    for (leaves, mcs, lr, ne), _ in top:
        if (leaves, mcs, lr, ne) == (63, 20, 0.05, 600): continue
        cfg = {**dv.TRAIN_CFG[d], "params": {**dv.TRAIN_CFG[d].get("params", {}), "num_leaves": leaves, "min_child_samples": mcs, "learning_rate": lr}}
        dd = []
        for s in (0, 1, 2):
            a = dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, s, cfg=cfg, n_estimators=ne)[0])["RMSE"]
            b = dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, s)[0])["RMSE"]
            dd.append(a - b)
        rows.append({"분야": nm, "잎": leaves, "최소표본": mcs, "lr": lr, "나무": ne, "시드": "0,1,2 (기본 대비 ΔRMSE)", "RMSE": round(np.mean(dd), 5), "MAE": " / ".join(f"{x:+.5f}" for x in dd)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "tune.csv", index=False, encoding="utf-8-sig")
    del df
