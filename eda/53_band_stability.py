# 보합 기준(±B) 바꾸기 + 연속된 날 예측 안정성. 저장된 회귀 예측(checkpoint) 사용, 축·수는 3진 LGB를 기준별 재학습
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
PRED = {"agri": "checkpoint2", "livestock": "checkpoint2", "fishery": "checkpoint3"}
BANDS = [0.01, 0.02, 0.03, 0.05]
rows, stab = [], []
cls = lambda v, b: np.where(v > b, 1, np.where(v < -b, -1, 0))
for d, nm in dv.cs.rm.NAMES.items():
    v = pd.read_parquet(ROOT / "eda" / "tables" / f"{PRED[d]}_pred_{d}.parquet")
    y, p = v.y.values.astype(float), v.pred.values
    for b in BANDS:
        big = np.abs(y) > b
        tc, pc = cls(y, b), cls(p, b)
        rows.append({"분야": nm, "보합 기준": f"±{b:.0%}", "방법": "회귀", "실제 보합 비율": round(np.mean(tc == 0), 3), "실제 상승 / 하락": f"{np.mean(tc == 1):.3f} / {np.mean(tc == -1):.3f}",
                     "방향정확도(|y|>기준)": round(np.mean(np.sign(p[big]) == np.sign(y[big])), 4), "3구간 정확도": round(np.mean(tc == pc), 4),
                     "예측 보합 비율": round(np.mean(pc == 0), 3), "오름·내림 표시 정확도 (표시 비율)": f"{np.mean(pc[pc != 0] == tc[pc != 0]):.3f} ({np.mean(pc != 0):.3f})"})
    # 연속된 날 안정성 (시계열별)
    v = v.sort_values(dv.cs.KEYS + ["sgg_nm", "date"])
    g = v.groupby(dv.cs.KEYS + ["sgg_nm"], observed=True, sort=False)
    gap = g.date.diff().dt.days
    pc2, tc2 = cls(v.pred.values, 0.02), cls(v.y.values, 0.02)
    prev_p = g.pred.shift(1).values; prev_pc = pd.Series(pc2, index=v.index).groupby([v[k] for k in dv.cs.KEYS + ["sgg_nm"]], observed=True, sort=False).shift(1).values
    prev_tc = pd.Series(tc2, index=v.index).groupby([v[k] for k in dv.cs.KEYS + ["sgg_nm"]], observed=True, sort=False).shift(1).values
    ok = (gap == 1).values
    stab.append({"분야": nm, "연속일 쌍": int(ok.sum()), "예측값 하루 차이 상관": round(np.corrcoef(v.pred.values[ok], prev_p[ok])[0, 1], 3),
                 "실제 y 하루 차이 상관": round(np.corrcoef(v.y.values[ok], g.y.shift(1).values[ok])[0, 1], 3),
                 "예측 라벨이 바뀐 비율 (±2%)": round(np.mean(pc2[ok] != prev_pc[ok]), 3), "실제 라벨이 바뀐 비율": round(np.mean(tc2[ok] != prev_tc[ok]), 3),
                 "오름↔내림 정반대로 바뀐 비율": round(np.mean((pc2[ok] * prev_pc[ok]) == -1), 4),
                 "예측 하루 변화 중앙값 |Δŷ|": round(np.median(np.abs(v.pred.values[ok] - prev_p[ok])), 4)})
    print(pd.DataFrame([r for r in rows if r["분야"] == nm]).to_string(index=False)); print(stab[-1], flush=True)
# 축·수: 3진 LGB를 보합 기준별로 재학습 (시드 42)
for d in ["livestock", "fishery"]:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); yv = df.y.values.astype(float); y = yv[te]
    X = dv.matrix(df, num, cats, tr, 42)
    for b in BANDS:
        yc = cls(yv, b) + 1
        P = lgb.LGBMClassifier(**{**dv.LGB_BASE, "random_state": 42}, objective="multiclass", num_class=3).fit(X[tr], yc[tr], categorical_feature=cats).predict_proba(X[te])
        tc, pc = cls(y, b), P.argmax(1) - 1; big = np.abs(y) > b
        dirn = np.where(P[:, 2] >= P[:, 0], 1, -1)
        rows.append({"분야": nm, "보합 기준": f"±{b:.0%}", "방법": "3진 분류 LGB", "실제 보합 비율": round(np.mean(tc == 0), 3),
                     "방향정확도(|y|>기준)": round(np.mean(dirn[big] == np.sign(y[big])), 4), "3구간 정확도": round(np.mean(tc == pc), 4),
                     "예측 보합 비율": round(np.mean(pc == 0), 3), "오름·내림 표시 정확도 (표시 비율)": f"{np.mean(pc[pc != 0] == tc[pc != 0]):.3f} ({np.mean(pc != 0):.3f})"})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "band_compare.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(stab).to_csv(ROOT / "eda" / "tables" / "pred_stability.csv", index=False, encoding="utf-8-sig")
