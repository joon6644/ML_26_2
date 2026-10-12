"""4단계: 사람이 내린 판단과 그 근거 — "점수가 올라도 믿을 수 있는 개선만 남겼다"

    python demo/4_judgment.py

  1) 채택·기각한 실험 (가설 → 결과 → 판단)        전체 기록: eda/derived_log.md
  2) 검증셋에 과적합되지 않았나: 검증 vs 테스트
  3) 해마다 성능이 유지되나: 연 단위 전진 검증 (매년 직전 해까지 학습 → 다음 해 예측)
  4) 운 좋은 시드가 아닌가: 시드 42·43·44
지표는 모두 소매 행 기준, 방향 정확도 = 실제 변화 ±1% 초과 행에서 오름·내림 적중률.
"""
import numpy as np
import pandas as pd

from shared import NAMES, P, ROOT, label

T = ROOT / "eda" / "tables"
DECISIONS = [
    ("농산물 일조·기온의 평년 대비", "시설 과채류는 일조·기온이 생육을 결정", "과채류 R² 0.458→0.481, 시드 3개 모두 개선", "채택"),
    ("수산물 지수 수준값→변화율 교체", "지수 수준은 해마다 올라 '연도 표시'가 됨", "R² 0.224→0.242, 시드 3개 모두 개선", "채택"),
    ("축산물 단조 제약", "되돌림 방향을 가설대로 고정하면 일반화", "RMSE −0.00072 (시드 잡음의 3배 초과)", "채택"),
    ("농산물 단조 제약", "같은 원리를 농산물에도", "R² 0.449→0.430 (공급 충격 땐 오른 가격이 더 오름)", "기각"),
    ("한우 주간 경락가 (482회 새로 수집)", "도매가 먼저 움직이고 소매가 따라감", "효과 없음 — 한우 소매가는 4주 안에 안 따라감", "기각"),
    ("큰 모델 (2000그루·255잎)", "용량을 키우면 더 정확", "R² 0.441→0.432, 과적합", "기각"),
    ("MLP (딥러닝)", "트리와 다른 오차 → 앙상블 다양성", "농산물 R² −1.96, 처음 보는 값에 불안정", "기각"),
    ("스태킹·가중 평균", "모델별 강점을 살린 조합", "2022년에 맞춘 조합이 2023~24에 안 옮겨짐", "기각"),
    ("하이퍼파라미터 튜닝", "기본값보다 나은 설정이 있을 것", "단일 시드 개선이 3시드에서 절반 이하, 기준(2σ) 미달", "기각 (기본값 유지)"),
    ("보합 기준 ±2% → ±1%", "기준을 넓히면 점수만 오름, 끊기는 지점 없음", "점수가 아니라 판단으로 ±1% 하나로 통일", "사람이 결정"),
    ("이상치: 하루짜리 튐 지우기", "잘못 기록된 값은 없는 셈 친다", "효과 없음 (차이가 시드 잡음 수준)", "기각"),
    ("이상치: 1.5 IQR 밖 전부 지우기", "통계적 이상치는 잡음", "효과 없음, 축산은 조사값 14%가 지워짐 (대부분 실제 변화)", "기각"),
    ("이상치: 하루짜리 튐 → 전날 값 대치", "잘못 기록됐을 가능성이 높으니 가격은 전날 그대로였다", "농산 검증·테스트 모두 소폭 개선, 축·수 동일", "채택 (세 분야 공통)"),
    ("앙상블 (LGB+XGB+Cat) → LightGBM 하나", "다른 모델을 섞으면 오차가 상쇄됨", "세 모델 오차 상관 0.95~0.99, 이득 농 R² +0.01~0.02·축·수 잡음 수준. '세 모델 합의' 신뢰도도 사실상 예측 크기와 같은 정보", "LightGBM 하나 (시스템 복잡도 대비 이득 제한적)"),
    ("결측: 평균 채움 → 아무 처리 안 함", "트리 모델은 '비어 있음' 자체를 신호로 쓸 수 있음", "6칸 중 5칸 RMSE 소폭 개선 (잡음 수준), 단계 하나 줄어 단순", "채택"),
    ("학습 처리 통일: 1~99% 자르기", "분야별로 다른 처리 대신 한 규칙", "농·축 유지, 수산 방향 정확도 −0.005 수준 (잡음 범위)", "채택 (설명 단순화)"),
]


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
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("\n■ 1. 채택·기각한 실험 (가설을 먼저 적고, 시드 잡음의 2배를 넘는 개선만 채택)\n")
    print(pd.DataFrame(DECISIONS, columns=["시도", "가설", "결과", "판단"]).to_string(index=False))

    print("\n■ 2. 검증(2023~24)에서 고른 구성이 처음 보는 테스트(2025~26)에서도 유지되나\n")
    e = pd.read_csv(T / "final_eval.csv")
    e = e[e.모델 == "LightGBM"].pivot(index="분야", columns="구간", values=["R²", "방향 정확도"])
    print(e.reindex(["농산물", "축산물", "수산물"]).to_string())

    print("\n■ 3. 연 단위 전진 검증 — 해마다 R²는 흔들려도 방향은 유지되나 (이전 구성 기준: 분야별 학습 처리·정제 없음·결측 평균 채움, LightGBM 예측)\n")
    print(walk_forward().to_string(index=False))

    print("\n■ 4. 시드 42·43·44 — 운 좋은 시드가 아닌가 (검증 2023~24, 이전 구성 기준, LightGBM 예측)\n")
    print(seeds().to_string(index=False))

    print("\n■ 참고: R² 상한 — 품목 공통 움직임을 완벽히 맞혀도 지역 잡음 때문에 농 0.71 / 축 0.47 / 수 0.27이 한계")
