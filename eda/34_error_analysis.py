# 농산물 오차 분석: 품목·월·지역/전국별로 오차 제곱합(SSE) 비중과 품목별 R², 공통 움직임 상한
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
d = sys.argv[1] if len(sys.argv) > 1 else "agri"; nm = dv.cs.rm.NAMES[d]
df = dv.build_all(d); tr, te = dv.masks(df)
num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]
p = dv.train_predict(df, d, num, cats, tr, te, 0)[0]
v = df.loc[te, dv.cs.KEYS + ["date", "sgg_nm", "is_nat", "food_group", "y"]].copy(); v["p"] = p
v["se"] = (v.y - v.p) ** 2
v["common"] = v.groupby(dv.cs.KEYS + ["date"]).y.transform("mean")
v["se_or"] = (v.y - v.common) ** 2
tot = v.se.sum(); sst = ((v.y - v.y.mean()) ** 2).sum()
g = v.groupby("item_nm").agg(행=("y", "size"), SSE비중=("se", lambda s: s.sum() / tot * 100), y분산=("y", "var"),
                             R2=("se", "sum"), 상한SSE=("se_or", "sum"))
sst_i = v.groupby("item_nm").y.apply(lambda s: ((s - s.mean()) ** 2).sum())
g["R2"] = 1 - g.R2 / sst_i; g["공통 상한 R²"] = 1 - g.상한SSE / sst_i
g = g.drop(columns="상한SSE").sort_values("SSE비중", ascending=False)
print("== 품목별 (SSE 비중 상위 25)"); print(g.head(25).round(3).to_string())
print("상위 10 품목 SSE 비중 합:", round(g.SSE비중.head(10).sum(), 1), "%")
v["ym"] = v.date.dt.to_period("M")
m = v.groupby("ym").agg(SSE비중=("se", lambda s: s.sum() / tot * 100), y평균=("y", "mean"), p평균=("p", "mean"))
print("== 월별"); print(m.round(3).to_string())
print("== 지역/전국:", v.groupby("is_nat").se.sum().div(tot).mul(100).round(1).to_dict())
g.to_csv(ROOT / "eda" / "tables" / f"error_by_item_{d}.csv", encoding="utf-8-sig")
