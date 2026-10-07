# 5라운드: 분야별 회귀 후보 6종 다중 시드 검증 (확인용 시드 42,0,1,2,3 / 서비스는 42)
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "50_diverse.py", encoding="utf-8").read().split("for d in DOMS:")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
SEEDS = [42, 0, 1, 2, 3]
CANDS = {"① LGB": ["lgb"], "② LGB+XGB": ["lgb", "xgb"], "③ LGB+Ridge": ["lgb", "ridge"], "④ LGB+XGB+Ridge": ["lgb", "xgb", "ridge"],
         "⑤ 부스팅3종": ["lgb", "xgb", "cat"], "⑥ 부스팅3종+Ridge": ["lgb", "xgb", "cat", "ridge"]}
SIG2 = {"농산물": 0.001, "축산물": 0.0004, "수산물": 0.0008}
rows, cost = [], []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); y = df.y.values[te].astype(float)
    ridge_p, tmr = fit_all(df, d, num, cats, tr, te, models=["ridge"]); ridge_p = ridge_p["ridge"]
    T = {"ridge": tmr["ridge"]}
    res = {}
    for s in SEEDS:
        SEED = s
        P, tm = fit_all(df, d, num, cats, tr, te, models=["lgb", "xgb", "cat"]); P["ridge"] = ridge_p
        for k, v in tm.items(): T.setdefault(k, []).append(v) if isinstance(T.get(k, []), list) else None
        for c, ms in CANDS.items():
            res.setdefault(c, []).append(dv.metrics(y, np.mean([P[m] for m in ms], axis=0)))
        print(nm, "시드", s, {c: round(res[c][-1]["RMSE"], 5) for c in CANDS}, flush=True)
    t_model = {k: (np.mean(v) if isinstance(v, list) else v) for k, v in T.items()}
    for c, ms in CANDS.items():
        R = res[c]; rm = np.array([r["RMSE"] for r in R]); r1 = np.array([r["RMSE"] for r in res["① LGB"]])
        diff = rm - r1
        rows.append({"분야": nm, "후보": c, "RMSE 평균": round(rm.mean(), 5), "RMSE 시드 표준편차": round(rm.std(ddof=1), 5),
                     "R² 평균": round(np.mean([r["R2"] for r in R]), 4), "방향정확도 평균": round(np.mean([r["방향정확도"] for r in R]), 4),
                     "3구간 평균": round(np.mean([r["3구간정확도"] for r in R]), 4),
                     "① 대비 ΔRMSE (평균, 시드별 일관)": f"{diff.mean():+.5f} ({(diff < 0).sum()}/{len(diff)} 시드 개선)",
                     "판정 (vs ①)": "-" if c == "① LGB" else ("좋아짐" if diff.mean() < -SIG2[nm] else ("나빠짐" if diff.mean() > SIG2[nm] else "차이 없음")),
                     "학습 시간(초)": round(sum(t_model[m] for m in ms), 1)})
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "final_verify.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([r for r in rows if r["분야"] == nm]).to_string(index=False), flush=True)
    del df
