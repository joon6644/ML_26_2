# 예측 크기 → "실제로 그 방향으로 1% 넘게 움직일 확률" (사용자 결정 2026-10-10)
# 정의 B: P( sign(ŷ) · y > 1%  |  |ŷ| ) — 예측 방향으로 실제 4주 평균이 1% 넘게 움직일 확률 (거의 안 움직이면 '틀림')
# 방법 (2026-10-12 변경): 선형회귀 확률 = a + b·ln|ŷ| (0~85%로 자름), 분야별, 검증(2023~24) 예측으로 맞추고 테스트(2025~26)에서 확인
#   예측 크기를 로그 눈금으로 보면 적중률이 거의 직선 → b·ln2 = '예측이 2배 커지면 확률이 몇 %p 오르나'.
#   이전 등위 회귀는 계단 모양이라(예측 크기가 조금 달라도 같은 확률, 경계에서 뜀) 바꿈. 로지스틱 회귀도 비교 → 테스트 Brier 같거나 선형이 약간 나음 (아래 비교표)
# 표시 규칙: 식 값을 1% 단위로(구간 표 없이), 최대 85% (테스트에서 높은 구간이 과대평가됨), 50% 이하면 "신뢰성이 낮으니 판단에 주의하세요"
# 출력: models/prob_{분야}.joblib (서비스·시연용 계수), eda/tables/direction_prob.csv (예측 크기별 확률), direction_prob_check.csv (테스트 확인),
#       eda/figures/direction_prob_curve.png (예측 크기 → 확률 곡선 + 테스트 실제 적중률)
import sys
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
T, P0, M, F = ROOT / "eda" / "tables", ROOT / "data" / "processed", ROOT / "models", ROOT / "eda" / "figures"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
CAP, STEP = 0.85, 0.01                                                   # 식 값 그대로(1% 단위), 최대 85%
sys.stdout.reconfigure(encoding="utf-8")


def load(d):
    v = pd.read_parquet(P0 / d / "model_frame_v1.parquet", columns=["date", "se_nm", "y"])
    v = v[(v.date >= "2023-01-01") & (v.date <= "2024-12-31") & v.y.notna()]
    t = pd.read_parquet(P0 / d / "model_input.parquet", columns=["split", "se_nm", "y"])
    t = t[t.split == "테스트"]
    pv, pt = np.load(T / f"final_val_preds_{d}.npz")["lgb"], np.load(T / f"final_test_preds_{d}.npz")["lgb"]
    rv, rt = (v.se_nm == "소매").values, (t.se_nm == "소매").values
    return v.y.values[rv], pv[rv], t.y.values[rt], pt[rt]


def hits(p, y):
    m = np.abs(p) > 0.01                                                  # 보합 예측(±1% 이내)은 방향을 말하지 않음
    return np.abs(p[m]), (np.sign(p[m]) * y[m] > 0.01).astype(float)


def prob(a, b, size):
    return np.clip(a + b * np.log(size), 0, 1)


def shown(pr):
    return np.minimum(np.round(pr / STEP) * STEP, CAP)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    grid, check, cmp_ = [], [], []
    for ax, (d, nm) in zip(axes, NAMES.items()):
        yv, pv, yt, pt = load(d)
        sv, hv = hits(pv, yv)
        st, ht = hits(pt, yt)
        lin = LinearRegression().fit(np.log(sv)[:, None], hv)
        a, b = float(lin.intercept_), float(lin.coef_[0])
        iso = IsotonicRegression(out_of_bounds="clip").fit(sv, hv)        # 이전 방식 (비교용)
        lg = LogisticRegression(C=1e6).fit(np.log(sv)[:, None], hv)      # 비교용
        cmp_.append({"분야": nm, "a": round(a, 3), "b": round(b, 3), "예측 2배 → 확률 +%p": round(b * np.log(2) * 100, 1),
                     "선형 Brier": round(float(np.mean((prob(a, b, st) - ht) ** 2)), 4),
                     "로지스틱 Brier": round(float(np.mean((lg.predict_proba(np.log(st)[:, None])[:, 1] - ht) ** 2)), 4),
                     "등위 회귀 Brier": round(float(np.mean((iso.predict(st) - ht) ** 2)), 4)})
        joblib.dump({"a": a, "b": b, "cap": CAP, "step": STEP, "fit_on": "검증 2023~24 (LightGBM 예측, 소매)"}, M / f"prob_{d}.joblib")
        for g in [0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            pr = float(prob(a, b, g))
            grid.append({"분야": nm, "예측 크기 |ŷ|": f"{g:.1%}", "확률 (보정값)": round(pr, 3), "표시": f"{shown(pr):.0%}" if pr > 0.5 else "신뢰성 낮음"})
        sh = shown(prob(a, b, st))
        for lab, m in [("50% 이하 (주의 문구)", sh <= 0.5)] + [(f"{x - 0.04:.0%}~{x:.0%}", (sh > x - 0.05 + 1e-9) & (sh <= x + 1e-9)) for x in np.arange(0.55, CAP + 1e-9, 0.05)]:   # 확인용으로만 5%p 구간에 묶음
            if m.any():
                check.append({"분야": nm, "표시": lab, "테스트 행 비율": round(float(m.mean()), 3), "실제 적중률": round(float(ht[m].mean()), 3)})
        # 그래프: 곡선(검증으로 맞춤) + 테스트 실제 적중률(예측 크기 구간별)
        x = np.geomspace(0.01, 0.30, 200)
        ax.plot(x * 100, np.minimum(prob(a, b, x), CAP) * 100, color="#d1495b", lw=2, label="표시 확률 (검증으로 맞춤, 최대 85%)")
        edges = np.quantile(st, np.linspace(0, 1, 13))
        bins = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, 11)
        mid = [np.median(st[bins == k]) * 100 for k in range(12)]
        act = [ht[bins == k].mean() * 100 for k in range(12)]
        ax.plot(mid, act, "o", color="#2e86ab", ms=6, label="테스트 실제 적중률 (2025~26)")
        ax.axhline(50, color="#999", lw=0.8, ls=":")
        ax.text(29, 51, "50% 이하 → 주의 문구", fontsize=8, color="#666", ha="right")
        ax.set_xlim(0, 30)                                                # 원본 눈금 → 식이 로그 곡선으로 보임
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_title(nm)
        ax.set_xlabel("예측 변화 크기 (절댓값)")
    axes[0].set_ylabel("예측 방향으로 1% 넘게 움직일 확률 (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(F / "direction_prob_curve.png", dpi=150)
    pd.DataFrame(grid).to_csv(T / "direction_prob.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(check).to_csv(T / "direction_prob_check.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("■ 선형 vs 로지스틱 vs 등위 회귀 (테스트 Brier 점수, 낮을수록 좋음)\n", pd.DataFrame(cmp_).to_string(index=False))
    print("\n■ 예측 크기별 확률 (검증으로 맞춤)\n", pd.DataFrame(grid).pivot(index="예측 크기 |ŷ|", columns="분야", values="표시").to_string())
    print("\n■ 테스트 확인: 표시한 확률 vs 실제 적중률\n", pd.DataFrame(check).to_string(index=False))
    print("\n그래프 저장: eda/figures/direction_prob_curve.png")
