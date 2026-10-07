# 7차: D18 주산지 기상 누적, D19 같은 계열 오늘 튐. 기준 = 기본 + 채택 + 분야별 학습 설정, LGB 단일 모델로 비교 (앙상블은 최종에)
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["agri", "livestock", "fishery"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df = dv.add_derived4(dv.add_derived(dv.cs.build(d)[0], d), d)
    tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    new = dv.DERIVED4[d]
    print(nm, "build", round(time.time() - t0), "s | 결측%", {c: round(df[c].isna().mean() * 100, 1) for c in new}, flush=True)
    run = lambda n, s: dv.metrics(y, dv.train_predict(df, d, n, cats, tr, te, s)[0])
    base = {0: run(num, 0)}
    V = {f"+ {c}": num + [c] for c in new}
    if len(new) > 1: V["+ 전부"] = num + new
    if d == "agri": V["+ 주산지 3개 (D18)"] = num + new[:3]
    for k, n in V.items():
        r = {0: run(n, 0)}
        if sig <= abs(r[0]["RMSE"] - base[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                if s not in base: base[s] = run(num, s)
                r[s] = run(n, s)
        ks = list(r)
        row = {"분야": nm, "변형": k, "시드": ",".join(map(str, ks))}
        for m in ["RMSE", "R2", "방향정확도"]:
            row[m] = round(np.mean([r[s][m] for s in ks]), 4); row[f"Δ{m}"] = round(np.mean([r[s][m] - base[s][m] for s in ks]), 5)
        dd = row["ΔRMSE"]
        row["판정"] = "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "batch7.csv", index=False, encoding="utf-8-sig")
    del df
