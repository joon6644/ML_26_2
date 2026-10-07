# 14차: M15 방향 3분류 (축·수 우선, 농 포함). 비교: 회귀(최종 구성 LGB 시드 0)의 부호 vs 분류. 확률 임계값별 표시 비율·정확도
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("dv", ROOT / "eda" / "24_derived.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
sys.stdout.reconfigure(encoding="utf-8")
DOMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["livestock", "fishery", "agri"]
rows = []
for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.final_frame(d); tr, te = dv.masks(df); yv = df.y.values; y = yv[te]
    cls = np.where(yv > .02, 2, np.where(yv < -.02, 0, 1))
    preds = []
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for s in range(NS):
        X = dv.matrix(df, num, cats, tr, s)
        m = lgb.LGBMClassifier(**{**dv.LGB_BASE, "objective": "multiclass", "num_class": 3, "random_state": s}).fit(X[tr], cls[tr], categorical_feature=cats)
        preds.append(m.predict_proba(X[te]))
    P = np.mean(preds, axis=0)
    big = np.abs(y) > .02
    reg = pd.read_parquet(ROOT / "eda" / "tables" / f"checkpoint2_pred_{d}.parquet").pred.values
    # 회귀 기준선
    for t in (0.02, 0.05):
        m_ = np.abs(reg) >= t
        rows.append({"분야": nm, "방법": f"회귀(최종) |ŷ|≥{t:.0%}", "표시 비율": round(m_.mean(), 3), "방향 정확도(|y|>2%)": round(np.mean(np.sign(reg[m_ & big]) == np.sign(y[m_ & big])), 3)})
    # 분류: 오름/내림 확률이 보합보다 크고 임계값 이상일 때 표시
    up, down = P[:, 2], P[:, 0]
    dirn = np.where(up >= down, 1, -1); conf = np.maximum(up, down)
    for t in (0.4, 0.5, 0.6, 0.7):
        m_ = conf >= t
        rows.append({"분야": nm, "방법": f"분류 max(P오름,P내림)≥{t}", "표시 비율": round(m_.mean(), 3), "방향 정확도(|y|>2%)": round(np.mean(dirn[m_ & big] == np.sign(y[m_ & big])), 3)})
    cl_pred = P.argmax(1); true = np.where(y > .02, 2, np.where(y < -.02, 0, 1))
    rows.append({"분야": nm, "방법": "분류 3구간 정확도 (argmax)", "표시 비율": 1.0, "방향 정확도(|y|>2%)": round(np.mean(cl_pred == true), 3)})
    regcls = np.where(reg > .02, 2, np.where(reg < -.02, 0, 1))
    rows.append({"분야": nm, "방법": "회귀 3구간 정확도", "표시 비율": 1.0, "방향 정확도(|y|>2%)": round(np.mean(regcls == true), 3)})
    for r in rows[-8:]: print(r, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / f"direction_clf_s{NS}.csv", index=False, encoding="utf-8-sig")
    np.save(ROOT / "eda" / "tables" / (f"direction_clf_proba_{d}.npy" if NS == 3 else f"direction_clf_proba_{d}_s{NS}.npy"), P)
    del df
