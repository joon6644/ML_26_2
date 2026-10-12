# [현재 결과] 지금 보고서·서비스 숫자를 만드는 스크립트 (eda/README.md)
# ★ 확정 구성 v3 성능표 (2026-10-10): LightGBM 하나 + 데이터 정제(하루짜리 튐 → 전날 값) + 타깃 1~99% 자르기 + 결측 처리 없음
# 검증 예측 = eda/tables/final_val_preds_{분야}.npz 의 lgb (eda/70·71과 같은 구성, ~2022-12-01 학습 → 2023~24)
# 테스트 예측 = eda/tables/final_test_preds_{분야}.npz (demo/0_train_models.py, ~2024-12-01 학습 → 2025~)
# 출력: eda/tables/final_eval.csv (베이스라인 vs LightGBM, RMSE·MAE·R²·방향 정확도) · final_eval_aux.csv (실제 변화 크기별 · 예측 크기별 방향 정확도)
# 이전 앙상블 성능표는 final_eval_ensemble.csv 로 보관
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T, P0 = ROOT / "eda" / "tables", ROOT / "data" / "processed"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
BINS = [(0.01, 0.02, "1~2%"), (0.02, 0.05, "2~5%"), (0.05, 0.10, "5~10%"), (0.10, 9.9, "10% 이상")]
sys.stdout.reconfigure(encoding="utf-8")


def metrics(y, p):
    mv = np.abs(y) > 0.01
    return {"RMSE": round(float(np.sqrt(np.mean((p - y) ** 2))), 4), "MAE": round(float(np.mean(np.abs(p - y))), 4),
            "R²": round(float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
            "방향 정확도": round(float(np.mean(np.sign(p[mv]) == np.sign(y[mv]))), 3) if p.any() else np.nan}


rows, aux = [], []
for d, nm in NAMES.items():
    v = pd.read_parquet(P0 / d / "model_frame_v1.parquet", columns=["date", "se_nm", "y"])
    v = v[(v.date >= "2023-01-01") & (v.date <= "2024-12-31") & v.y.notna()]
    t = pd.read_parquet(P0 / d / "model_input.parquet", columns=["split", "se_nm", "y"])
    t = t[t.split == "테스트"]
    for sname, E, f in [("검증 (2023~24)", v, f"final_val_preds_{d}.npz"), ("테스트 (2025~26)", t, f"final_test_preds_{d}.npz")]:
        p = np.load(T / f)["lgb"]
        assert len(p) == len(E), (d, sname, len(p), len(E))
        r = (E.se_nm == "소매").values
        y, p = E.y.values.astype(float)[r], p[r]
        for model, pp in [("베이스라인", np.zeros(len(y))), ("LightGBM", p)]:
            rows.append({"분야": nm, "구간": sname, "모델": model, **metrics(y, pp)})
        ok = np.sign(p) == np.sign(y)
        for kind, a, m_extra in [("실제 변화 크기", np.abs(y), True), ("예측 크기", np.abs(p), np.abs(y) > 0.01)]:
            for lo, hi, lb in BINS:
                m = (a > lo) & (a <= hi)
                aux.append({"분야": nm, "구간": sname, "구분": kind, "항목": lb, "비율": round(float(m.mean()), 3),
                            "방향 정확도": round(float(ok[m & m_extra].mean()), 3)})
pd.DataFrame(rows).to_csv(T / "final_eval.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(aux).to_csv(T / "final_eval_aux.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).fillna("–").to_string(index=False))
print(pd.DataFrame(aux).to_string(index=False))
