# 연 단위 전진 검증 (테스트 2025~ 제외): Y년 평가 = 2016-01-01 ~ (Y-1)-12-01 학습 → Y년 전체 예측
# 학습 끝을 12-01로 둬 학습 타깃(향후 28일)이 평가 연도로 넘어가지 않게 함 (기존 검증 분할과 같은 규칙)
# 최종 구성(LGB+XGB+Cat 균등 평균, 분야별 학습 설정·단조 제약, 시드 42) + 베이스라인 1 + 공통 움직임 상한. 연도별 예측은 저장해 재사용
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "eda" / "tables"
YEARS = {"agri": range(2019, 2025), "fishery": range(2019, 2025), "livestock": range(2023, 2025)}   # 축산물 지역 행은 2022년부터
rows = []
for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); yv = df.y.values.astype(float)
    cfg = dv.TRAIN_CFG[d]
    mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
    for Y in YEARS[d]:
        t = time.time()
        tr = ((df.date >= "2016-01-01") & (df.date <= f"{Y - 1}-12-01")).values
        te = ((df.date >= f"{Y}-01-01") & (df.date <= f"{Y}-12-31")).values
        f = OUT / f"wf_preds_{d}_{Y}.npz"
        if f.exists():
            P = dict(np.load(f))
        else:
            yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
            P = fit3(df, d, num, cats, tr, te, yt, 42, mono)
            np.savez(f, **P)
        y = yv[te]; mean = np.mean([P[k] for k in ["lgb", "xgb", "cat"]], axis=0)
        r = dv.metrics(y, mean); b = dv.metrics(y, np.expm1(df.dev_last.fillna(0).values[te]))
        Lm = cls(mean); t3 = cls(y); big = t3 != 0
        L = np.column_stack([cls(P[k]) for k in ["lgb", "xgb", "cat"]]); agree = (L == L[:, :1]).all(1); hi = agree & (Lm != 0)
        # 공통 움직임 상한 (지역 행): 같은 시계열·날짜 지역 평균을 완벽히 맞힐 때 R²
        reg = te & (df.is_nat.values == 0)
        g = df.loc[reg, dv.cs.KEYS + ["date", "y"]]
        orc = g.groupby(dv.cs.KEYS + ["date"], observed=True).y.transform("mean").values; yr = g.y.values
        ceil = 1 - np.sum((yr - orc) ** 2) / np.sum((yr - yr.mean()) ** 2) if reg.any() else np.nan
        rr = dv.metrics(yr, mean[reg[te]]) if reg.any() else {"R2": np.nan}
        rows.append({"분야": nm, "평가 연도": Y, "학습 기간": f"2016~{Y - 1}", "학습 행": int(tr.sum()), "평가 행": int(te.sum()),
                     "RMSE": round(r["RMSE"], 5), "R²": round(r["R2"], 4), "베이스라인1 R²": round(b["R2"], 4), "방향 정확도": round(r["방향정확도"], 4),
                     "3구간": round(np.mean(Lm == t3), 4), "높음 비율": round(hi.mean(), 3), "높음 방향 정확도": round(np.mean(Lm[hi & big] == t3[hi & big]), 4),
                     "낮음 비율": round((~agree).mean(), 3), "지역 행 R²": round(rr["R2"], 4), "공통 움직임 상한": round(ceil, 3),
                     "실제 |y| 평균": round(float(np.mean(np.abs(y))), 4), "초": round(time.time() - t)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "walk_forward.csv", index=False, encoding="utf-8-sig")
    del df
print(pd.DataFrame(rows).to_string(index=False))
