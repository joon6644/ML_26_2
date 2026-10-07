# (1) 검증 2년 분기별 성능 (학습 끝에서 멀어질수록?) (2) 연도별 가격 수준 변화 vs 소비자물가 상승률 (2016~2024만)
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
q_rows, y_rows = [], []
for d in ["agri", "livestock", "fishery"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    if d == "agri":
        P = dict(np.load(ROOT / "eda" / "tables" / "preds_agri_s42.npz")); P = {k: P[k] for k in ["lgb", "xgb", "cat"]}
    else:
        mono = {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
        P = fit3(df, d, num, cats, tr, te, yt, 42, mono)
        np.savez(ROOT / "eda" / "tables" / f"final_preds_{d}_s42.npz", **P)
    mean = np.mean(list(P.values()), axis=0); Lm = cls(mean); t3 = cls(y); big = t3 != 0
    L = np.column_stack([cls(v) for v in P.values()]); hi = (L == L[:, :1]).all(1) & (Lm != 0)
    dates = df.date.values[te]; q = pd.PeriodIndex(pd.to_datetime(dates), freq="Q").astype(str)
    for qq in sorted(set(q)):
        m = q == qq; b = m & big; h = m & hi & big
        r = dv.metrics(y[m], mean[m])
        q_rows.append({"분야": nm, "분기": qq, "행": int(m.sum()), "RMSE": round(r["RMSE"], 4), "R²": round(r["R2"], 3), "방향 정확도": round(r["방향정확도"], 3),
                       "높음 비율": round(np.mean(hi[m]), 3), "높음 방향 정확도": round(np.mean(np.sign(mean[h]) == np.sign(y[h])), 3),
                       "실제 y 평균": round(y[m].mean(), 4), "실제 y 표준편차": round(y[m].std(), 4), "예측 평균": round(mean[m].mean(), 4)})
    # (2) 연도별 가격 수준: 전국 시계열별 연평균 가격의 전년 대비 log 변화 → 중앙값. 소비자물가는 같은 해 연평균
    f = pd.read_parquet(cs_path := (dv.cs.P / d / "features.parquet"), columns=dv.cs.KEYS + ["date", "price_kg", "cpi_총지수", "cpi_식료품"])
    f = f[(f.date >= "2015-01-01") & (f.date <= "2024-12-31")]
    f["year"] = f.date.dt.year
    a = f.groupby(dv.cs.KEYS + ["year"], observed=True).price_kg.mean().rename("p").reset_index()
    a["prev"] = a.groupby(dv.cs.KEYS, observed=True).p.shift(1)
    a = a[(a.year == a.groupby(dv.cs.KEYS, observed=True).year.shift(1) + 1)] if False else a
    a["chg"] = np.log(a.p / a.prev)
    g = a.dropna(subset=["chg"]).groupby("year").chg.agg(["median", "mean", "count"])
    cpi = f.groupby("year")[["cpi_총지수", "cpi_식료품"]].mean()
    cpi_chg = np.log(cpi / cpi.shift(1))
    for yr in g.index:
        y_rows.append({"분야": nm, "연도": yr, "품목 가격 전년비 중앙값 %": round(np.expm1(g.loc[yr, "median"]) * 100, 1),
                       "품목 가격 전년비 평균 %": round(np.expm1(g.loc[yr, "mean"]) * 100, 1), "시계열 수": int(g.loc[yr, "count"]),
                       "소비자물가 총지수 상승률 %": round(np.expm1(cpi_chg.loc[yr, "cpi_총지수"]) * 100, 1) if yr in cpi_chg.index else None,
                       "소비자물가 식료품 상승률 %": round(np.expm1(cpi_chg.loc[yr, "cpi_식료품"]) * 100, 1) if yr in cpi_chg.index else None})
    del df, f
Q = pd.DataFrame(q_rows); Y = pd.DataFrame(y_rows)
Q.to_csv(ROOT / "eda" / "tables" / "valid_by_quarter.csv", index=False, encoding="utf-8-sig"); Y.to_csv(ROOT / "eda" / "tables" / "price_vs_cpi.csv", index=False, encoding="utf-8-sig")
print(Q.to_string(index=False)); print(Y.to_string(index=False))
for nm in Y.분야.unique():
    t = Y[Y.분야 == nm].dropna()
    print(nm, "연도별 품목 가격 중앙값 상승률 vs 총지수 상승률 상관", round(t["품목 가격 전년비 중앙값 %"].corr(t["소비자물가 총지수 상승률 %"]), 2),
          "| vs 식료품", round(t["품목 가격 전년비 중앙값 %"].corr(t["소비자물가 식료품 상승률 %"]), 2),
          "| 누적(2016→2024) 품목 중앙값 %", round((np.prod(1 + t["품목 가격 전년비 중앙값 %"] / 100) - 1) * 100, 1),
          "총지수 %", round((np.prod(1 + t["소비자물가 총지수 상승률 %"] / 100) - 1) * 100, 1), "식료품 %", round((np.prod(1 + t["소비자물가 식료품 상승률 %"] / 100) - 1) * 100, 1))
