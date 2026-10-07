# 1차 파생변수 실험: 기본 + 각 파생변수 하나씩 / 기본 + 전부. 시드 0, 애매하면 1·2 추가
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d, nm in dv.cs.rm.NAMES.items():
    t = time.time()
    df = dv.add_derived(dv.cs.build(d)[0], d); tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); sig = dv.cs.SIG[nm]
    print(nm, "build", round(time.time() - t), "s | 결측%", {c: round(df[c].isna().mean() * 100, 1) for c in dv.DERIVED[d]}, flush=True)
    def run(n, s): return dv.metrics(y, dv.cs.fit(df, n, cats, tr, te, s))
    base = {0: run(num, 0)}
    V = {f"+ {c}": num + [c] for c in dv.DERIVED[d]}
    V["+ 전부"] = num + dv.DERIVED[d]
    for k, n in V.items():
        r = {0: run(n, 0)}
        if sig <= abs(r[0]["RMSE"] - base[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                base.setdefault(s, run(num, s)); r[s] = run(n, s)
        ks = list(r)
        row = {"분야": nm, "변형": k, "시드": ",".join(map(str, ks))}
        for mtr in ["RMSE", "R2", "방향정확도", "3구간정확도", "상승재현율"]:
            row[mtr] = round(np.mean([r[s][mtr] for s in ks]), 4)
            row[f"Δ{mtr}"] = round(np.mean([r[s][mtr] - base[s][mtr] for s in ks]), 5)
        dd = row["ΔRMSE"]
        row["판정"] = "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")
        rows.append(row); print({k2: row[k2] for k2 in ["변형", "시드", "RMSE", "ΔRMSE", "R2", "ΔR2", "방향정확도", "Δ방향정확도", "판정"]}, flush=True)
    rows.append({"분야": nm, "변형": "기본", "시드": "0", **{m: round(base[0][m], 4) for m in ["RMSE", "R2", "방향정확도", "3구간정확도", "상승재현율"]}})
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "derived_batch1.csv", index=False, encoding="utf-8-sig")
    del df
