# 중간 점검: 기본 + 채택 파생변수 + 분야별 학습 설정 + 채택 앙상블. 확정 포맷 표 (베이스라인 vs 모델)
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
spec = importlib.util.spec_from_file_location("en", ROOT / "eda" / "32_ensemble.py")
src = open(ROOT / "eda" / "32_ensemble.py", encoding="utf-8").read().split("for d in DOMS:")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
TAG = sys.argv[1] if len(sys.argv) > 1 else "checkpoint"
DOMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["livestock", "fishery", "agri"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df, num, cats = dv.final_frame(d); tr, te = dv.masks(df); yv = df.y.values; y = yv[te]
    dev = np.expm1(df.dev_last.fillna(0).values[te])
    lgbs = [dv.train_predict(df, d, num, cats, tr, te, s)[0] for s in range(5)]
    p = np.mean(lgbs, axis=0)
    if dv.ENSEMBLE[d] == "lgb5+xgb+cat":
        X = dv.matrix(df, num, cats, tr, 0); yc = clip_y(d, yv, tr)
        p = (p + xgb_pred(d, X, yc, tr, te, cats) + cat_pred(d, X, yc, tr, te, cats)) / 3
    nat = df.is_nat.values[te] == 1
    p5 = np.mean(lgbs, axis=0)
    cands = [("베이스라인 1 (오늘 가격 유지)", dev), ("LightGBM 단일 (시드 0)", lgbs[0]), ("LightGBM 5시드", p5)]
    if dv.ENSEMBLE[d] != "lgb5": cands.append((f"LGB5+XGB+Cat", p))
    for lab, pp in cands:
        r = {"분야": nm, "모델": lab, "피처 수": "-" if lab.startswith("베이스") else len(num) + len(cats), **{k: round(v, 4) for k, v in dv.metrics(y, pp).items()},
             "R² 전국 행": round(dv.metrics(y[nat], pp[nat])["R2"], 4), "R² 지역 행": round(dv.metrics(y[~nat], pp[~nat])["R2"], 4)}
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / f"{TAG}.csv", index=False, encoding="utf-8-sig")
    out = df.loc[te, dv.cs.KEYS + ["sgg_nm", "date", "is_nat", "food_group", "y"]].copy(); out["pred"] = p; out.attrs = {}
    out.to_parquet(ROOT / "eda" / "tables" / f"{TAG}_pred_{d}.parquet", index=False)
    print(nm, round(time.time() - t0), "s", flush=True)
    del df
