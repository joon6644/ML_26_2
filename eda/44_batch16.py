# 16차: D31 수출 증가율, D32 산지 단가 전년비 (수산물). 기준 = 최종 구성의 LGB 단일(단조 제약·Huber), 새 피처는 단조 제약 없음
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
d, nm = "fishery", "수산물"
df = dv.add_derived9(dv.build_all(d), d); tr, te = dv.masks(df); y = df.y.values[te]
num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
print("결측%", {c: round(df[c].isna().mean() * 100, 1) for c in ["export_yoy", "catch_unit_yoy"]}, flush=True)
gim = (df.item_nm == "김").values[te]
def run(n, s):
    p = dv.train_predict(df, d, n, cats, tr, te, s)[0]
    r = dv.metrics(y, p); r["김 R²"] = dv.metrics(y[gim], p[gim])["R2"]; return r
rows = []
B = {0: run(num, 0)}
for add in [["export_yoy"], ["catch_unit_yoy"], ["export_yoy", "catch_unit_yoy"]]:
    R = {0: run(num + add, 0)}
    if sig <= abs(R[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
        for s in (1, 2):
            B.setdefault(s, run(num, s)); R[s] = run(num + add, s)
    ks = list(R); dd = np.mean([R[s]["RMSE"] - B[s]["RMSE"] for s in ks])
    row = {"변형": "+ " + ", ".join(add), "시드": ",".join(map(str, ks)), **{k: round(np.mean([R[s][k] for s in ks]), 4) for k in ["RMSE", "R2", "방향정확도", "3구간정확도", "김 R²"]},
           "ΔRMSE": round(dd, 5), "판정": "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")}
    rows.append(row); print(row, flush=True)
rows.insert(0, {"변형": "기준", "시드": "0", **{k: round(B[0][k], 4) for k in ["RMSE", "R2", "방향정확도", "3구간정확도", "김 R²"]}}); print(rows[0])
pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "batch16.csv", index=False, encoding="utf-8-sig")
