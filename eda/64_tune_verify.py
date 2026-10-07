# 7라운드 검증: 모델별 순위 r위 설정 앙상블 vs 기본 앙상블 (시드 42·0·1). 1위가 기준 미달이면 2위·3위 순으로
import importlib.util, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ts", ROOT / "eda" / "63_tune_search.py"); ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
dv = ts.dv
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
SEEDS = [42, 0, 1]
SIG2 = {"농산물": 0.001, "축산물": 0.0004, "수산물": 0.0008}
S = pd.read_csv(ROOT / "eda" / "tables" / "tune_search.csv")
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in ts.LEVEL}
    rank = {m: S[(S.분야 == nm) & (S.모델 == m)].sort_values("RMSE").설정.map(json.loads).tolist() for m in ["lgb", "xgb", "cat"]}
    cache = {}
    def pred(model, hp, s):
        key = (model, json.dumps(hp, sort_keys=True), s)
        if key not in cache:
            if d == "agri" and hp == ts.DEFAULT[model] and s in (42, 0, 1):
                cache[key] = np.load(ROOT / "eda" / "tables" / f"preds_agri_s{s}.npz")[model]
            else:
                cache[key] = ts.fit_one(model, hp, df, d, num, cats, tr, te, yt, s, mono)
        return cache[key]
    def ens(hps, s): return np.mean([pred(m, hps[m], s) for m in ["lgb", "xgb", "cat"]], axis=0)
    base = {s: dv.metrics(y, ens(ts.DEFAULT, s)) for s in SEEDS}
    adopted = None
    for r in range(3):
        hps = {m: (rank[m][r] if r < len(rank[m]) else ts.DEFAULT[m]) for m in ["lgb", "xgb", "cat"]}
        if all(hps[m] == ts.DEFAULT[m] for m in hps):
            print(nm, f"{r + 1}위: 세 모델 모두 기본값 → 건너뜀", flush=True); continue
        t = time.time()
        res = {s: dv.metrics(y, ens(hps, s)) for s in SEEDS}
        diff = np.array([res[s]["RMSE"] - base[s]["RMSE"] for s in SEEDS])
        ok = diff.mean() < -SIG2[nm] and (diff < 0).sum() >= 2
        rows.append({"분야": nm, "후보": f"{r + 1}위 조합", "설정": json.dumps(hps, ensure_ascii=False), "기본 RMSE": round(np.mean([base[s]["RMSE"] for s in SEEDS]), 5),
                     "튜닝 RMSE": round(np.mean([res[s]["RMSE"] for s in SEEDS]), 5), "기본 R²": round(np.mean([base[s]["R2"] for s in SEEDS]), 4),
                     "튜닝 R²": round(np.mean([res[s]["R2"] for s in SEEDS]), 4), "ΔRMSE (시드별)": " / ".join(f"{x:+.5f}" for x in diff),
                     "판정": "채택" if ok else "기준 미달", "초": round(time.time() - t)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "tune_verify.csv", index=False, encoding="utf-8-sig")
        if ok:
            adopted = hps; break
    if adopted is None:
        rows.append({"분야": nm, "후보": "결론", "설정": "기본값 유지", "판정": "3위까지 모두 기준 미달"}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "tune_verify.csv", index=False, encoding="utf-8-sig")
    del df
