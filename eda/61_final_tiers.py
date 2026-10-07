# 확정 구성(시드 42)의 신뢰도 등급별 성능: 높음(같은 방향 만장일치) / 보통(모두 보합) / 낮음(엇갈림)
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for d in ["agri", "livestock", "fishery"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    cfg = dv.TRAIN_CFG[d]
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
    if d == "agri":
        P = dict(np.load(ROOT / "eda" / "tables" / "preds_agri_s42.npz")); P = {k: P[k] for k in ["lgb", "xgb", "cat"]}
    else:
        mono = {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
        P = fit3(df, d, num, cats, tr, te, yt, 42, mono)
    mean = np.mean(list(P.values()), axis=0); Lm = cls(mean); t3 = cls(y); big = t3 != 0
    L = np.column_stack([cls(v) for v in P.values()]); agree = (L == L[:, :1]).all(1)
    tiers = {"높음 (같은 방향 만장일치)": agree & (Lm != 0), "보통 (모두 보합)": agree & (Lm == 0), "낮음 (엇갈림)": ~agree, "전체": np.ones_like(agree)}
    for k, m in tiers.items():
        b = m & big
        rows.append({"분야": nm, "신뢰도": k, "비율": round(m.mean(), 3),
                     "표시 라벨(평균) = 실제 구간": round(np.mean(Lm[m] == t3[m]), 3),
                     "방향 정확도 (실제 |y|>2% 행)": round(np.mean(np.sign(mean[b]) == np.sign(y[b])), 3) if b.any() else None,
                     "실제로 ±2% 넘게 움직인 비율": round(np.mean(big[m]), 3),
                     "평균 |예측|": round(np.mean(np.abs(mean[m])), 3), "평균 |실제|": round(np.mean(np.abs(y[m])), 3),
                     "MAE": round(np.mean(np.abs(mean[m] - y[m])), 4)})
    r = dv.metrics(y, mean); rows.append({"분야": nm, "신뢰도": f"참고: 전체 R² {r['R2']:.3f}, RMSE {r['RMSE']:.4f}"})
    # 낮음 행 상세: 평균 라벨 구성
    low = ~agree
    rows.append({"분야": nm, "신뢰도": "  낮음 중 평균 라벨 구성 (오름/보합/내림)", "비율": f"{np.mean(Lm[low] == 1):.2f} / {np.mean(Lm[low] == 0):.2f} / {np.mean(Lm[low] == -1):.2f}"})
    del df
R = pd.DataFrame(rows); R.to_csv(ROOT / "eda" / "tables" / "final_tiers.csv", index=False, encoding="utf-8-sig")
print(R.fillna("").to_string(index=False))
