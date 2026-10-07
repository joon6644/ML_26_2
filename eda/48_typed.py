# 2라운드 A: 변수 성격별 파생변수 그룹 실험. 기준 = 현재 최종 구성의 LGB 단일(분야별 학습 설정). 시드 0, 애매하면 1·2
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df, G = dv.add_typed(dv.build_all(d), d); tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    lv = dv.typed_groups(d, nm)["G1"]
    print(nm, "build", round(time.time() - t0), "s", {g: len(v) for g, v in G.items()}, flush=True)
    V = {}
    if G["G1"]:
        V["+ G1 지수 변화율"] = num + G["G1"]
        V["G1′ 지수 수준 → 변화율 교체"] = [c for c in num if c not in lv] + G["G1"]
    if G["G2"]: V["+ G2 물량 전년 동기 대비"] = num + G["G2"]
    if G["G4"]: V["+ G4 해양·양식장 평년 대비"] = num + G["G4"]
    V["+ 전부"] = num + sum(G.values(), [])
    run = lambda n, s: dv.metrics(y, dv.train_predict(df, d, n, cats, tr, te, s)[0])
    B = {0: run(num, 0)}
    rows.append({"분야": nm, "변형": "기준", "시드": "0", **{k: round(B[0][k], 4) for k in ["RMSE", "R2", "방향정확도"]}})
    for k, n in V.items():
        R = {0: run(n, 0)}
        if sig <= abs(R[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                if s not in B: B[s] = run(num, s)
                R[s] = run(n, s)
        ks = list(R); dd = np.mean([R[s]["RMSE"] - B[s]["RMSE"] for s in ks])
        row = {"분야": nm, "변형": k, "피처 수": len(n) + len(cats), "시드": ",".join(map(str, ks)), **{m: round(np.mean([R[s][m] for s in ks]), 4) for m in ["RMSE", "R2", "방향정확도"]},
               "ΔRMSE": round(dd, 5), "ΔR2": round(np.mean([R[s]["R2"] - B[s]["R2"] for s in ks]), 4), "시드별": " / ".join(f"{R[s]['RMSE'] - B[s]['RMSE']:+.5f}" for s in ks),
               "판정": "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")}
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "typed.csv", index=False, encoding="utf-8-sig")
    del df
