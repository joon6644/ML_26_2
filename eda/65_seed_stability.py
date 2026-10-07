# 최종 설정을 시드 42·43·44로: 성능·신뢰도 등급이 시드에 따라 얼마나 흔들리는지
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
    labs = {}
    for s in (42, 43, 44):
        t = time.time()
        if d == "agri" and s == 42:
            P = dict(np.load(ROOT / "eda" / "tables" / "preds_agri_s42.npz")); P = {k: P[k] for k in ["lgb", "xgb", "cat"]}
        elif (ROOT / "eda" / "tables" / f"final_preds_{d}_s{s}.npz").exists():
            P = dict(np.load(ROOT / "eda" / "tables" / f"final_preds_{d}_s{s}.npz"))
        else:
            P = fit3(df, d, num, cats, tr, te, yt, s, mono)
            np.savez(ROOT / "eda" / "tables" / f"final_preds_{d}_s{s}.npz", **P)
        mean = np.mean(list(P.values()), axis=0); Lm = cls(mean); t3 = cls(y); big = t3 != 0
        L = np.column_stack([cls(v) for v in P.values()]); agree = (L == L[:, :1]).all(1); hi = agree & (Lm != 0)
        labs[s] = (Lm, np.where(hi, 2, np.where(agree, 1, 0)))
        r = dv.metrics(y, mean)
        rows.append({"분야": nm, "시드": s, "RMSE": round(r["RMSE"], 5), "R²": round(r["R2"], 4), "방향 정확도": round(r["방향정확도"], 4), "3구간": round(np.mean(Lm == t3), 4),
                     "높음 비율": round(hi.mean(), 3), "높음 방향 정확도": round(np.mean(Lm[hi & big] == t3[hi & big]), 4), "낮음 비율": round((~agree).mean(), 3), "초": round(time.time() - t)})
        print(rows[-1], flush=True)
    # 시드 간 같은 행에 같은 라벨·같은 신뢰도가 나오는 비율
    for a, b in [(42, 43), (42, 44), (43, 44)]:
        rows.append({"분야": nm, "시드": f"{a} vs {b} 일치율", "3구간": round(np.mean(labs[a][0] == labs[b][0]), 4), "높음 비율": round(np.mean(labs[a][1] == labs[b][1]), 4)})
    del df
R = pd.DataFrame(rows); R.to_csv(ROOT / "eda" / "tables" / "seed_stability.csv", index=False, encoding="utf-8-sig")
print(R.fillna("").to_string(index=False))
