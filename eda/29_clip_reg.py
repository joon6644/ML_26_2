# 4차: 타깃 극단값 처리(M1·M1b·M1c) × 규제(M2) × 최근 가중(M3). 피처 = 기본 + 채택 누적. 시드 0 (최종 후보는 시드 추가)
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mr", ROOT / "eda" / "28_model_reg.py")
sys.stdout.reconfigure(encoding="utf-8")
src = open(ROOT / "eda" / "28_model_reg.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)   # dv, BASEP, matrix 재사용
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(dv.cs.rm.NAMES)
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
CLIPS = {"clip 없음": None, "1~99%": (0.01, 0.99), "2.5~97.5%": (0.025, 0.975), "5~95%": (0.05, 0.95), "10~90%": (0.10, 0.90)}
REG = {"기본": {}, "R1 잎31·최소500": {"num_leaves": 31, "min_child_samples": 500}, "R2 잎15·최소2000·800그루": {"num_leaves": 15, "min_child_samples": 2000, "n_estimators": 800},
       "Huber δ0.05": {"objective": "huber", "alpha": 0.05}}
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df = dv.add_derived2(dv.add_derived(dv.cs.build(d)[0], d), d)
    tr, te = dv.masks(df); y = df.y.values
    num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]
    X = matrix(df, num, cats, tr)
    yr = df.date.dt.year.values[tr]
    combos = [(c, "기본", None) for c in CLIPS] + [(c, r, None) for c in ["5~95%"] for r in list(REG)[1:]] + [("clip 없음", "Huber δ0.05", None)] + \
             [("5~95%", "기본", "half2"), ("5~95%", "기본", "linear")]
    for cname, rname, wmode in combos:
        res = []
        for s in SEEDS:
            t = time.time()
            yt = y[tr] if CLIPS[cname] is None else np.clip(y[tr], *np.nanquantile(y[tr], CLIPS[cname]))
            w = None if wmode is None else ((yr - 2015).astype(float) if wmode == "linear" else 0.5 ** ((2022 - yr) / 2.0))
            m = lgb.LGBMRegressor(**{**BASEP, **REG[rname], "random_state": s}).fit(X[tr], yt, sample_weight=w, categorical_feature=cats)
            res.append(dv.metrics(y[te], m.predict(X[te])))
        r = {"분야": nm, "clip": cname, "모델": rname, "가중": wmode or "-", "시드": ",".join(map(str, SEEDS)),
             **{k: round(np.mean([x[k] for x in res]), 4) for k in res[0]}, "초": round(time.time() - t)}
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "clip_reg.csv", index=False, encoding="utf-8-sig")
    del df, X
