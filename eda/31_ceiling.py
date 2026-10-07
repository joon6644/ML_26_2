# 천장 진단: 검증 y의 분산 중 '시계열(품목·품종·등급·소매/중도매)×날짜 공통 움직임'이 설명하는 몫
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d, nm in dv.cs.rm.NAMES.items():
    df = dv.cs.build(d)[0]; tr, te = dv.masks(df)
    v = df.loc[te, dv.cs.KEYS + ["date", "y", "is_nat", "sgg_nm"]].copy()
    reg = v[v.is_nat == 0]
    common = reg.groupby(dv.cs.KEYS + ["date"]).y.transform("mean")
    n = reg.groupby(dv.cs.KEYS + ["date"]).y.transform("count")
    tot = np.var(v.y)
    r2_common_reg = 1 - np.mean((reg.y - common) ** 2) / np.var(reg.y)
    r2_common_reg_multi = 1 - np.mean(((reg.y - common) ** 2)[n >= 3]) / np.var(reg.y[n >= 3])
    rows.append({"분야": nm, "검증 행": len(v), "지역 행": len(reg), "같은 시계열·날짜의 평균 지역 수": round(n.mean(), 1),
                 "지역 행: 공통 움직임이 설명하는 몫 (오라클 R²)": round(r2_common_reg, 3),
                 "지역 3곳 이상인 경우": round(r2_common_reg_multi, 3)})
    print(rows[-1], flush=True); del df
pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "ceiling.csv", index=False, encoding="utf-8-sig")
