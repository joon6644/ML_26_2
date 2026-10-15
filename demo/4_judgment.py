"""4단계: 사람이 내린 판단과 그 근거 — "점수가 올라도 믿을 수 있는 개선만 남겼다"

    python demo/4_judgment.py

  1) 검증셋에 과적합되지 않았나: 검증 vs 테스트
  2) 해마다 성능이 유지되나: 연 단위 전진 검증 (매년 직전 해까지 학습 → 다음 해 예측)
  3) 운 좋은 시드가 아닌가: 시드 42·43·44
채택·기각한 실험과 근거는 eda/derived_log.md.
지표는 모두 소매 행 기준, 방향 정확도 = 실제 변화 ±1% 초과 행에서 오름·내림 적중률.
"""
import numpy as np
import pandas as pd

from shared import NAMES, P, ROOT, label, show

T = ROOT / "eda" / "tables"


def _score(y, P3):
    p = P3[:, 0]                                   # LightGBM 예측 (최종 구성 v3은 LightGBM 하나)
    mv = np.abs(y) > 0.01
    return {"R²": round(float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
            "방향 정확도": round(float(np.mean(np.sign(p[mv]) == np.sign(y[mv]))), 3)}


def walk_forward():
    f = T / "walk_forward_retail.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for d, years in [("agri", range(2019, 2025)), ("livestock", range(2023, 2025)), ("fishery", range(2019, 2025))]:
        df = pd.read_parquet(P / d / "model_frame_v1.parquet", columns=["date", "y", "se_nm"])
        for Y in years:
            m = ((df.date >= f"{Y}-01-01") & (df.date <= f"{Y}-12-31")).values
            r = df.se_nm.values[m] == "소매"
            z = np.load(T / f"wf_preds_{d}_{Y}.npz")
            P3 = np.column_stack([z[k] for k in ["lgb", "xgb", "cat"]])[r]
            rows.append({"분야": NAMES[d], "평가 연도": Y, "학습 기간": f"2016~{Y - 1}", **_score(df.y.values[m][r].astype(float), P3)})
    out = pd.DataFrame(rows)
    out.to_csv(f, index=False, encoding="utf-8-sig")
    return out


def seeds():
    f = T / "seed_stability_retail.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for d in ["agri", "livestock", "fishery"]:
        df = pd.read_parquet(P / d / "model_frame_v1.parquet", columns=["date", "y", "se_nm"])
        m = ((df.date >= "2023-01-01") & (df.date <= "2024-12-31")).values
        r = df.se_nm.values[m] == "소매"
        for s in (42, 43, 44):
            z = np.load(T / ("preds_agri_s42.npz" if (d, s) == ("agri", 42) else f"final_preds_{d}_s{s}.npz"))
            P3 = np.column_stack([z[k] for k in ["lgb", "xgb", "cat"]])[r]
            rows.append({"분야": NAMES[d], "시드": s, **_score(df.y.values[m][r].astype(float), P3)})
    out = pd.DataFrame(rows)
    out.to_csv(f, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    print("\n■ 1. 검증(2023~24)에서 고른 구성이 처음 보는 테스트(2025~26)에서도 유지되나\n")
    e = pd.read_csv(T / "final_eval.csv")
    e = e[e.모델 == "LightGBM"].pivot(index="분야", columns="구간", values=["R²", "방향 정확도"])
    e.columns = [f"{m} {span.split()[0]}" for m, span in e.columns]       # ('R²', '검증 (2023~24)') → 'R² 검증'
    show(e.reindex(["농산물", "축산물", "수산물"]), index=True)

    print("\n■ 2. 연 단위 전진 검증 — 해마다 R²는 흔들려도 방향은 유지되나 (이전 구성 기준: 분야별 학습 처리·정제 없음·결측 평균 채움, LightGBM 예측)\n")
    show(walk_forward())

    print("\n■ 3. 시드 42·43·44 — 운 좋은 시드가 아닌가 (검증 2023~24, 이전 구성 기준, LightGBM 예측)\n")
    show(seeds())

    print("\n■ 참고: R² 상한 — 품목 공통 움직임을 완벽히 맞혀도 지역 잡음 때문에 농 0.71 / 축 0.47 / 수 0.27이 한계")
