# 투표 신뢰도: 부스팅 3종(LGB·XGB·Cat) vs LGB·XGB·Ridge. 시드 42. 표시값 = 세 모델 평균
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "50_diverse.py", encoding="utf-8").read().split("for d in DOMS:")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
cls = lambda v: np.where(v > .02, 1, np.where(v < -.02, -1, 0))
rows = []
for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); y = df.y.values[te].astype(float)
    P = dict(np.load(ROOT / "eda" / "tables" / f"preds_{d}_s42.npz"))
    need = [m for m in ["lgb", "xgb", "cat", "ridge"] if m not in P]
    if need:
        SEED = 42
        P.update(fit_all(df, d, num, cats, tr, te, models=need)[0])
        np.savez(ROOT / "eda" / "tables" / f"preds_{d}_s42.npz", **P)
    t = cls(y); big = t != 0
    for name, ms in [("부스팅 3종 (LGB·XGB·Cat)", ["lgb", "xgb", "cat"]), ("LGB·XGB·Ridge", ["lgb", "xgb", "ridge"])]:
        mean = np.mean([P[m] for m in ms], axis=0); Lm = cls(mean)
        L = np.column_stack([cls(P[m]) for m in ms]); agree = (L == L[:, :1]).all(1); hi = agree & (Lm != 0)
        met = dv.metrics(y, mean)
        rows.append({"분야": nm, "투표자": name, "평균 RMSE": round(met["RMSE"], 5), "평균 R²": round(met["R2"], 4), "평균 라벨 3구간 정확도": round(np.mean(Lm == t), 4),
                     "만장일치 비율": round(agree.mean(), 3), "높음(오름·내림) 비율": round(hi.mean(), 3),
                     "높음: 방향 정확도": round(np.mean(Lm[hi & big] == t[hi & big]), 4), "높음: 라벨=실제 구간": round(np.mean(Lm[hi] == t[hi]), 4),
                     "일치 시 3구간": round(np.mean(Lm[agree] == t[agree]), 4), "불일치 시 3구간": round(np.mean(Lm[~agree] == t[~agree]), 4)})
        print(rows[-1], flush=True)
    del df
R = pd.DataFrame(rows); R.to_csv(ROOT / "eda" / "tables" / "vote_ridge.csv", index=False, encoding="utf-8-sig")
print(R.to_string(index=False))
