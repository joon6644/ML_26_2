# 13차: M14 단조 제약 (방향 점검에서 '일치' 확인된 피처만). 기준 = 기본 + 채택 + 분야별 학습 설정, LGB
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
chk = pd.read_csv(ROOT / "eda" / "tables" / "direction_check.csv")
EXTRA_DIR = {"agri": {"whsl_gap_anom": 1, "wx_rain_14d": 1}}   # 채택 파생변수의 사전 가설 방향 (D4 +, D6 +)
rows = []
for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    df = dv.build_all(d); tr, te = dv.masks(df); yv = df.y.values; y = yv[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    ok = chk[(chk.분야 == nm) & (chk.판정 == "일치")]
    mono = {r.feature: (1 if r._4 == "+" else -1) for r in ok.itertuples()} if "_4" in ok.columns else {}
    mono = {r["feature"]: (1 if r["기대 방향"] == "+" else -1) for _, r in ok.iterrows()}
    mono.update({k: v for k, v in EXTRA_DIR.get(d, {}).items() if k in num})
    mono = {k: v for k, v in mono.items() if k in num}
    print(nm, "단조 제약", len(mono), "개:", mono, flush=True)
    cfg = dv.TRAIN_CFG[d]
    def run(s, use):
        X = dv.matrix(df, num, cats, tr, s)
        cons = [mono.get(c, 0) if use else 0 for c in X.columns]
        yt = yv[tr] if not cfg.get("clip") else np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"]))
        m = lgb.LGBMRegressor(**{**dv.LGB_BASE, **cfg.get("params", {}), "random_state": s, "monotone_constraints": cons,
                                 "monotone_constraints_method": "advanced"}).fit(X[tr], yt, categorical_feature=cats)
        return dv.metrics(y, m.predict(X[te]))
    B, M = {0: run(0, False)}, {0: run(0, True)}
    if sig <= abs(M[0]["RMSE"] - B[0]["RMSE"]) <= 3 * sig:
        for s in (1, 2): B[s] = run(s, False); M[s] = run(s, True)
    dd = np.mean([M[s]["RMSE"] - B[s]["RMSE"] for s in M])
    for lab, R in [("기준", B), ("M14 단조 제약", M)]:
        r = {"분야": nm, "변형": lab, "시드": ",".join(map(str, R)), "제약 수": len(mono) if lab != "기준" else 0, **{k: round(np.mean([R[s][k] for s in R]), 4) for k in R[0]}}
        rows.append(r); print(r, flush=True)
    rows[-1]["ΔRMSE"] = round(dd, 5)
    rows[-1]["판정"] = "차이 없음" if abs(dd) < sig or (len(M) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")
    print("ΔRMSE", rows[-1]["ΔRMSE"], rows[-1]["판정"], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "monotone.csv", index=False, encoding="utf-8-sig")
    del df
