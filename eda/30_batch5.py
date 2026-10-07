# 5차: D15·D16·D17 파생변수 + M6(2022 조기 종료로 나무 수) + M7(log1p 타깃). 기준 = 기본 + 채택 누적, 분야별 학습 설정(TRAIN_CFG)
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
rows = []


def es_rounds(df, d, num, cats, tr, seed):
    """2016~2021 학습, 2022 평가로 조기 종료 → 최적 나무 수."""
    yr = df.date.dt.year.values
    tr_in, va = tr & (yr <= 2021), tr & (yr == 2022)
    cfg = dv.TRAIN_CFG[d]
    X = dv.matrix(df, num, cats, tr, seed)
    y = df.y.values
    yt = y[tr_in] if not cfg.get("clip") else np.clip(y[tr_in], *np.nanquantile(y[tr], cfg["clip"]))
    m = lgb.LGBMRegressor(**{**dv.LGB_BASE, **cfg.get("params", {}), "n_estimators": 3000, "random_state": seed})
    m.fit(X[tr_in], yt, eval_set=[(X[va], y[va])], eval_metric="l2", categorical_feature=cats, callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.best_iteration_


for d in DOMS:
    nm = dv.cs.rm.NAMES[d]; t0 = time.time()
    df = dv.add_derived3(dv.add_derived2(dv.add_derived(dv.cs.build(d)[0], d), d), d)
    tr, te = dv.masks(df); y = df.y.values[te]
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
    print(nm, "build", round(time.time() - t0), "s | 결측%", {c: round(df[c].isna().mean() * 100, 1) for c in dv.DERIVED3}, flush=True)

    def run(variant, s):
        if variant.startswith("+"):
            add = dv.DERIVED3 if variant == "+ 3개 전부" else [variant[2:]]
            return dv.metrics(y, dv.train_predict(df, d, num + add, cats, tr, te, s)[0])
        if variant.startswith("M6"):
            n = es_rounds(df, d, num, cats, tr, s)
            out = dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, s, n_estimators=n)[0]); out["나무수"] = n
            return out
        if variant.startswith("M7"):
            p = dv.train_predict(df, d, num, cats, tr, te, s, target=np.log1p(df.y.values))[0]
            return dv.metrics(y, np.expm1(p))
        return dv.metrics(y, dv.train_predict(df, d, num, cats, tr, te, s)[0])

    base = {0: run("기준", 0)}
    rows.append({"분야": nm, "변형": "기준 (기본+채택, 분야별 학습 설정)", "시드": "0", **{k: round(v, 4) for k, v in base[0].items()}})
    print(rows[-1], flush=True)
    for v in [f"+ {c}" for c in dv.DERIVED3] + ["+ 3개 전부", "M6 2022 조기 종료", "M7 log1p 타깃"]:
        r = {0: run(v, 0)}
        if sig <= abs(r[0]["RMSE"] - base[0]["RMSE"]) <= 3 * sig:
            for s in (1, 2):
                if s not in base: base[s] = run("기준", s)
                r[s] = run(v, s)
        ks = list(r)
        row = {"분야": nm, "변형": v, "시드": ",".join(map(str, ks))}
        for k in ["RMSE", "R2", "방향정확도", "3구간정확도", "상승재현율"]:
            row[k] = round(np.mean([r[s][k] for s in ks]), 4); row[f"Δ{k}"] = round(np.mean([r[s][k] - base[s][k] for s in ks]), 5)
        if "나무수" in r[0]: row["나무수"] = ",".join(str(r[s]["나무수"]) for s in ks)
        dd = row["ΔRMSE"]
        row["판정"] = "차이 없음" if abs(dd) < sig or (len(ks) == 3 and abs(dd) <= 2 * sig) else ("나빠짐" if dd > 0 else "좋아짐")
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "batch5.csv", index=False, encoding="utf-8-sig")
    del df
