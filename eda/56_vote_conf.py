# 신뢰도 방식 비교: (가) 평균 회귀 라벨 + 3진 분류 일치 vs (나) 회귀 3종 개별 라벨 투표 일치. 시드 42
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
    need = [m for m in ["lgb", "xgb", "cat"] if m not in P]
    if need:
        SEED = 42
        P.update(fit_all(df, d, num, cats, tr, te, models=need)[0])
    t = cls(y); big = t != 0
    mean = (P["lgb"] + P["xgb"] + P["cat"]) / 3; Lm = cls(mean)
    L = np.column_stack([cls(P[m]) for m in ["lgb", "xgb", "cat"]])
    clf = np.load(ROOT / "eda" / "tables" / f"classify_proba_{d}.npz")["p_l"].argmax(1) - 1
    schemes = {
        "(가) 평균 라벨 + 3진 분류 일치": (Lm == clf),
        "(나) 회귀 3종 만장일치": (L == L[:, :1]).all(1),
        "(나′) 회귀 3종 중 2개 이상이 평균 라벨과 같음": ((L == Lm[:, None]).sum(1) >= 2),
    }
    for name, agree in schemes.items():
        hi = agree & (Lm != 0)          # 신뢰도 높음: 일치 + 오름/내림
        lo = ~agree
        rows.append({"분야": nm, "방식": name, "일치 비율": round(agree.mean(), 3),
                     "높음(오름·내림) 비율": round(hi.mean(), 3),
                     "높음: 방향 정확도(실제 |y|>2%)": round(np.mean(Lm[hi & big] == t[hi & big]), 4),
                     "높음: 라벨=실제 구간": round(np.mean(Lm[hi] == t[hi]), 4),
                     "일치 시 3구간 정확도": round(np.mean(Lm[agree] == t[agree]), 4),
                     "불일치 시 3구간 정확도": round(np.mean(Lm[lo] == t[lo]), 4) if lo.any() else None})
        print(rows[-1], flush=True)
    rows.append({"분야": nm, "방식": "참고: 평균 라벨 전체", "일치 비율": 1.0, "높음(오름·내림) 비율": round((Lm != 0).mean(), 3),
                 "높음: 방향 정확도(실제 |y|>2%)": round(np.mean(Lm[(Lm != 0) & big] == t[(Lm != 0) & big]), 4),
                 "높음: 라벨=실제 구간": round(np.mean(Lm[Lm != 0] == t[Lm != 0]), 4), "일치 시 3구간 정확도": round(np.mean(Lm == t), 4)})
    del df
R = pd.DataFrame(rows); R.to_csv(ROOT / "eda" / "tables" / "vote_conf.csv", index=False, encoding="utf-8-sig")
print(R.fillna("").to_string(index=False))
