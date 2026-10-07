# 12차: 시장 구성 효과 D28·D29·D30 (농·수, 지역 행). 기준 = 기본 + 채택 + 분야별 학습 설정, LGB
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fishery", "agri"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df = dv.add_derived8(dv.build_all(d), d); tr, te = dv.masks(df); y = df.y.values[te]
    reg = df.is_nat.values[te] == 0
    print(nm, "build", round(time.time() - t0), "s | 지역 행 결측%", {c: round(df.loc[df.is_nat == 0, c].isna().mean() * 100, 1) for c in dv.DERIVED8},
          "| comp_effect 분포", df.comp_effect.describe().round(3).to_dict(), flush=True)
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    def run(n, s):
        p = dv.train_predict(df, d, n, cats, tr, te, s)[0]
        r = dv.metrics(y, p); r["R2 지역"] = dv.metrics(y[reg], p[reg])["R2"]; return r
    B = {0: run(num, 0)}
    V = {f"+ {c}": num + [c] for c in dv.DERIVED8}; V["+ 3개 전부"] = num + dv.DERIVED8
    for k, n in V.items():
        R = {0: run(n, 0)}
        if sig <= abs(R[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                B.setdefault(s, run(num, s)); R[s] = run(n, s)
        ks = list(R); dd = np.mean([R[s]["RMSE"] - B[s]["RMSE"] for s in ks])
        row = {"분야": nm, "변형": k, "시드": ",".join(map(str, ks)), **{m: round(np.mean([R[s][m] for s in ks]), 4) for m in ["RMSE", "R2", "R2 지역", "방향정확도"]},
               "ΔRMSE": round(dd, 5), "ΔR2": round(np.mean([R[s]["R2"] - B[s]["R2"] for s in ks]), 4),
               "판정": "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")}
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "market_comp.csv", index=False, encoding="utf-8-sig")
    rows.append({"분야": nm, "변형": "기준", "시드": "0", **{m: round(B[0][m], 4) for m in ["RMSE", "R2", "R2 지역", "방향정확도"]}})
    del df
