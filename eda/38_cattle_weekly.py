# 10차: D24·D25 한우 주간 경락가 (축산물). 기준 = 기본 + 채택 + 분야별 학습 설정, LGB. 시드 0, 애매하면 1·2
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
d, nm = "livestock", "축산물"
df = dv.add_derived6(dv.build_all(d), d); tr, te = dv.masks(df); y = df.y.values[te]
num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
beef = df.item_nm.isin(["소", "수입 소고기"]).values[te]
print("결측% (소·수입소고기 행)", {c: round(df.loc[df.item_nm.isin(["소", "수입 소고기"]), c].isna().mean() * 100, 1) for c in ["cattle_w_mom4", "cattle_w_gap_anom"]}, flush=True)
def run(n, s):
    p = dv.train_predict(df, d, n, cats, tr, te, s)[0]
    r = dv.metrics(y, p); r["소·수입소 R²"] = dv.metrics(y[beef], p[beef])["R2"]; return r
rows = []
B = {0: run(num, 0)}
for add in [["cattle_w_mom4"], ["cattle_w_gap_anom"], ["cattle_w_mom4", "cattle_w_gap_anom"]]:
    R = {0: run(num + add, 0)}
    if sig <= abs(R[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
        for s in (1, 2):
            B.setdefault(s, run(num, s)); R[s] = run(num + add, s)
    ks = list(R)
    row = {"변형": "+ " + ", ".join(add), "시드": ",".join(map(str, ks)), **{k: round(np.mean([R[s][k] for s in ks]), 4) for k in R[0]},
           "ΔRMSE": round(np.mean([R[s]["RMSE"] - B[s]["RMSE"] for s in ks]), 5), "Δ소·수입소 R²": round(np.mean([R[s]["소·수입소 R²"] - B[s]["소·수입소 R²"] for s in ks]), 4)}
    rows.append(row); print(row, flush=True)
rows.insert(0, {"변형": "기준", "시드": "0", **{k: round(v, 4) for k, v in B[0].items()}})
pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "cattle_weekly.csv", index=False, encoding="utf-8-sig")
print(rows[0])
