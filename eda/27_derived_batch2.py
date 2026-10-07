# 2차 파생변수 실험: (기본 + 채택 누적) + 각 2차 파생변수 하나씩 / 전부, 그리고 M1(학습 y 1~99% clip). 시드 0, 애매하면 1·2 추가
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d, nm in dv.cs.rm.NAMES.items():
    t = time.time()
    df = dv.add_derived2(dv.add_derived(dv.cs.build(d)[0], d), d); tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    print(nm, "build", round(time.time() - t), "s | 결측%", {c: round(df[c].isna().mean() * 100, 1) for c in dv.DERIVED2[d]}, flush=True)
    y_orig = df.y.values.copy()
    lo, hi = np.nanquantile(y_orig[tr], [0.01, 0.99])
    def run(n, s, clip=False):
        df["y"] = np.where(tr, np.clip(y_orig, lo, hi), y_orig) if clip else y_orig
        out = dv.metrics(y, dv.cs.fit(df, n, cats, tr, te, s))
        df["y"] = y_orig
        return out
    base = {0: run(num, 0)}
    V = {f"+ {c}": num + [c] for c in dv.DERIVED2[d]}
    V["+ 전부"] = num + dv.DERIVED2[d]
    V["M1 학습 y 1~99% clip"] = num
    for k, n in V.items():
        cl = k.startswith("M1")
        r = {0: run(n, 0, cl)}
        if sig <= abs(r[0]["RMSE"] - base[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                base.setdefault(s, run(num, s)); r[s] = run(n, s, cl)
        ks = list(r)
        row = {"분야": nm, "변형": k, "시드": ",".join(map(str, ks))}
        for mtr in ["RMSE", "R2", "방향정확도", "3구간정확도", "상승재현율"]:
            row[mtr] = round(np.mean([r[s][mtr] for s in ks]), 4)
            row[f"Δ{mtr}"] = round(np.mean([r[s][mtr] - base[s][mtr] for s in ks]), 5)
        dd = row["ΔRMSE"]
        row["판정"] = "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")
        rows.append(row); print({k2: row[k2] for k2 in ["변형", "시드", "RMSE", "ΔRMSE", "R2", "ΔR2", "방향정확도", "Δ방향정확도", "판정"]}, flush=True)
    rows.append({"분야": nm, "변형": "기본", "시드": "0", **{m: round(base[0][m], 4) for m in ["RMSE", "R2", "방향정확도", "3구간정확도", "상승재현율"]}})
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "derived_batch2.csv", index=False, encoding="utf-8-sig")
    del df
