# 19차: M18 학습 타깃 잡음 줄이기 (지역 행 개별 변동을 λ배). 최종 학습 설정 위 LGB, 시드 0 → 애매하면 1·2
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.build_all(d); tr, te = dv.masks(df); yv = df.y.values; y = yv[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    reg = df.is_nat.values == 0
    common = df.y.where(reg).groupby([df[k] for k in dv.cs.KEYS] + [df.date], observed=True).transform("mean").values
    def run(s, lam):
        t = yv.copy()
        if lam is not None:
            m = tr & reg & np.isfinite(common)
            t[m] = common[m] + lam * (yv[m] - common[m])
        return dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, s, target=t)[0])
    B = {0: run(0, None)}
    for lam in (0.75, 0.5):
        R = {0: run(0, lam)}
        if sig <= abs(R[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                if s not in B: B[s] = run(s, None)
                R[s] = run(s, lam)
        ks = list(R); dd = np.mean([R[s]["RMSE"] - B[s]["RMSE"] for s in ks])
        r = {"분야": nm, "λ": lam, "시드": ",".join(map(str, ks)), "R2 기준": round(np.mean([B[s]["R2"] for s in ks]), 4), "R2 M18": round(np.mean([R[s]["R2"] for s in ks]), 4),
             "방향 기준": round(np.mean([B[s]["방향정확도"] for s in ks]), 4), "방향 M18": round(np.mean([R[s]["방향정확도"] for s in ks]), 4),
             "ΔRMSE": round(dd, 5), "시드별": " / ".join(f"{R[s]['RMSE'] - B[s]['RMSE']:+.5f}" for s in ks),
             "판정": "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")}
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "denoise.csv", index=False, encoding="utf-8-sig")
    del df
