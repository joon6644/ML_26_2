# 8차 (농산물): D20 일조 이상, D21 기온 이상, M11 계열별 모델. 기준 = 기본 + 채택, LGB 단일
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
d, nm = "agri", "농산물"
df = dv.add_derived5(dv.add_derived(dv.cs.build(d)[0], d), d); tr, te = dv.masks(df); y = df.y.values[te]
num, cats = dv.base_sets(nm); num = num + dv.ADOPTED[d]; sig = dv.cs.SIG[nm]
veg = df.food_group.isin(["과채류"]).values[te]
rows = []
def ev(p, lab, seeds="0"):
    r = {"변형": lab, "시드": seeds, **{k: round(v, 4) for k, v in dv.metrics(y, p).items()},
         "과채류 R²": round(dv.metrics(y[veg], p[veg])["R2"], 4)}
    rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "batch8.csv", index=False, encoding="utf-8-sig")
    return r
P = {s: dv.train_predict(df, d, num, cats, tr, te, s)[0] for s in (0,)}
b = ev(P[0], "기준")
for add in [["wx_sun_14d_anom"], ["wx_temp_14d_anom"], ["wx_sun_14d_anom", "wx_temp_14d_anom"]]:
    p0 = dv.train_predict(df, d, num + add, cats, tr, te, 0)[0]
    r = ev(p0, "+ " + ", ".join(add))
    if sig <= abs(r["RMSE"] - b["RMSE"]) <= 3 * sig:
        ps = [p0] + [dv.train_predict(df, d, num + add, cats, tr, te, s)[0] for s in (1, 2)]
        if 1 not in P:
            for s in (1, 2): P[s] = dv.train_predict(df, d, num, cats, tr, te, s)[0]
        dd = np.mean([dv.metrics(y, ps[i])["RMSE"] - dv.metrics(y, P[i])["RMSE"] for i in range(3)])
        rows.append({"변형": "  ↳ 시드 0~2 ΔRMSE", "RMSE": round(dd, 5)}); print(rows[-1], flush=True)
# M11: 계열별 별도 모델 (같은 피처, 계열 안에서 학습)
t = time.time()
p = np.full(te.sum(), np.nan)
fg = df.food_group.values
for g in pd.unique(fg):
    mtr, mte = tr & (fg == g), te & (fg == g)
    if mte.sum() == 0 or mtr.sum() < 1000:
        continue
    sub = df[(fg == g)].reset_index(drop=True); sub.attrs = df.attrs.copy()
    sub.attrs["scope"] = {k: v[fg == g] for k, v in df.attrs.get("scope", {}).items()}
    str_, ste = mtr[fg == g], mte[fg == g]
    p[np.where(fg[te] == g)[0]] = dv.train_predict(sub, d, num, cats, str_, ste, 0)[0]
miss = np.isnan(p); p[miss] = P[0][miss]
ev(p, f"M11 계열별 모델 (계열 {len(pd.unique(fg))}개, 결측 {miss.mean() * 100:.1f}%는 기준으로 채움, {round(time.time() - t)}초)")
