# 앱 활용 분석 (사전 기록한 계획대로): 예측 크기 임계값별 표시 비율·방향 정확도, 품목별 신뢰도
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
TAG = sys.argv[1] if len(sys.argv) > 1 else "checkpoint2"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
rows, items = [], []
for d, nm in NAMES.items():
    v = pd.read_parquet(ROOT / "eda" / "tables" / f"{TAG}_pred_{d}.parquet")
    for t in (0.0, 0.02, 0.05, 0.10):
        m = np.abs(v.pred) >= t
        sub = v[m]
        big = sub[np.abs(sub.y) > 0.02]
        rows.append({"분야": nm, "표시 임계값 |예측|≥": f"{t:.0%}", "표시 비율": round(m.mean(), 3),
                     "방향 정확도 (실제 |y|>2% 행)": round(np.mean(np.sign(big.pred) == np.sign(big.y)), 3) if len(big) else np.nan,
                     "방향 정확도 (표시 행 전체, 실제 부호)": round(np.mean(np.sign(sub.pred) == np.sign(sub.y)), 3) if len(sub) else np.nan,
                     "표시 행 중 실제 |y|>2% 비율": round(len(big) / max(len(sub), 1), 3)})
    def r2(g): return 1 - ((g.y - g.pred) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()
    g = v.groupby("item_nm").apply(lambda g: pd.Series({"행": len(g), "R²": r2(g),
                                                       "방향 정확도": np.mean(np.sign(g.pred[np.abs(g.y) > .02]) == np.sign(g.y[np.abs(g.y) > .02])),
                                                       "|예측|≥5% 표시 비율": np.mean(np.abs(g.pred) >= .05),
                                                       "|예측|≥5% 방향 정확도": np.mean(np.sign(g.pred[np.abs(g.pred) >= .05]) == np.sign(g.y[np.abs(g.pred) >= .05])) if (np.abs(g.pred) >= .05).any() else np.nan}),
                                       include_groups=False)
    g.insert(0, "분야", nm); items.append(g.reset_index())
R = pd.DataFrame(rows); I = pd.concat(items, ignore_index=True).round(3)
R.to_csv(ROOT / "eda" / "tables" / "app_threshold.csv", index=False, encoding="utf-8-sig")
I.to_csv(ROOT / "eda" / "tables" / "app_item_reliability.csv", index=False, encoding="utf-8-sig")
print(R.to_string(index=False)); print(I.sort_values(["분야", "R²"], ascending=[True, False]).to_string(index=False))
