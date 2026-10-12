# [현재 결과] 지금 보고서·서비스 숫자를 만드는 스크립트 (eda/README.md)
# 예측 크기 → "실제로 그 방향으로 1% 넘게 움직일 확률" (사용자 결정 2026-10-10)
# 정의 B: P( sign(ŷ) · y > 1%  |  |ŷ| ) — 예측 방향으로 실제 4주 평균이 1% 넘게 움직일 확률 (거의 안 움직이면 '틀림')
# 방법 (2026-10-12 변경): 선형회귀 확률 = a + b·ln|ŷ| (0 ~ 분야별 상한으로 자름), 분야별
#   평가용: 검증(2023~24) 예측으로만 맞추고 테스트(2025~26)에서 확인 — 성능 보고 숫자
#   상한: 분야별로 예측 크기 상위 1%의 실제 적중률 (사용자 결정 2026-10-12) — 곡선이 그 높이에 닿으면 그 뒤는 수평. 가장 큰 예측들이 실제로 맞은 만큼만 말함
#   서비스용: 학습에 안 쓴 예측 중 정답이 나온 것 전부(검증 2023~24 + 서비스 모델의 2025~ 예측)로 맞춤 — 카드에 쓰는 계수.
#     실제 운영에서 정답을 못 보는 건 마지막 4주뿐이므로. 2025까지로 맞추고 2026으로 확인했을 때 검증만 쓴 식보다 농·수에서 나음
#     (평균 차이 농 3.5→3.2%p, 수 1.4→1.2%p). 기간에 따른 흔들림(2026 축산 등)은 어떤 식으로도 미리 반영할 수 없어 감수
#   예측 크기를 로그 눈금으로 보면 적중률이 거의 직선 → b·ln2 = '예측이 2배 커지면 확률이 몇 %p 오르나'.
#   이전 등위 회귀는 계단 모양이라(예측 크기가 조금 달라도 같은 확률, 경계에서 뜀) 바꿈. 로지스틱 회귀도 비교 → 테스트 Brier 같거나 선형이 약간 나음 (아래 비교표)
# 표시 규칙: 식 값을 1% 단위로(구간 표 없이), 분야별 상한까지, 50% 이하면 "신뢰성이 낮으니 판단에 주의하세요"
# 출력: models/prob_{분야}.joblib (서비스·시연용 계수), eda/tables/direction_prob_check.csv (평가: 테스트 확인),
#       eda/figures/direction_prob_curve.png (카드 규칙 그대로의 곡선 + 학습 밖 예측의 실제 적중률)
import sys
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
T, P0, M, F = ROOT / "eda" / "tables", ROOT / "data" / "processed", ROOT / "models", ROOT / "eda" / "figures"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
STEP, TOP, PER = 0.01, 0.99, 1000                                         # 식 값 그대로(1% 단위) / 상한 = 상위 1% 실제 적중률 / 그림: 1,000행씩 묶어 전부
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


def shown(pr, cap):
    return np.minimum(np.round(pr / STEP) * STEP, cap)


def top_cap(size, hit):
    """분야별 상한: 예측 크기 상위 1%가 실제로 맞은 비율, 1% 단위 내림."""
    return float(np.floor(hit[size >= np.quantile(size, TOP)].mean() * 100) / 100)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    check, cmp_, coef = [], [], []
    for ax, (d, nm) in zip(axes, NAMES.items()):
        yv, pv, yt, pt = load(d)
        sv, hv = hits(pv, yv)
        st, ht = hits(pt, yt)
        # 평가용: 검증만으로 맞춤 → 테스트 확인
        lin = LinearRegression().fit(np.log(sv)[:, None], hv)
        a, b = float(lin.intercept_), float(lin.coef_[0])
        iso = IsotonicRegression(out_of_bounds="clip").fit(sv, hv)        # 이전 방식 (비교용)
        lg = LogisticRegression(C=1e6).fit(np.log(sv)[:, None], hv)      # 비교용
        cmp_.append({"분야": nm, "선형 Brier": round(float(np.mean((prob(a, b, st) - ht) ** 2)), 4),
                     "로지스틱 Brier": round(float(np.mean((lg.predict_proba(np.log(st)[:, None])[:, 1] - ht) ** 2)), 4),
                     "등위 회귀 Brier": round(float(np.mean((iso.predict(st) - ht) ** 2)), 4)})
        cap_e = top_cap(sv, hv)                                           # 평가용 상한도 검증만으로
        sh = shown(prob(a, b, st), cap_e)
        for lab, m in [("50% 이하 (주의 문구)", sh <= 0.5)] + [(f"{x - 0.04:.0%}~{x:.0%}", (sh > x - 0.05 + 1e-9) & (sh <= x + 1e-9)) for x in np.arange(0.55, cap_e + 0.05, 0.05)]:   # 확인용으로만 5%p 구간에 묶음
            if m.sum() > 100:
                check.append({"분야": nm, "표시": lab, "테스트 행 비율": round(float(m.mean()), 3), "실제 적중률": round(float(ht[m].mean()), 3)})
        # 서비스용: 정답이 나온 학습 밖 예측 전부
        sa, sb = np.polyfit(np.log(np.r_[sv, st]), np.r_[hv, ht], 1)[::-1]
        cap = top_cap(np.r_[sv, st], np.r_[hv, ht])
        joblib.dump({"a": float(sa), "b": float(sb), "cap": cap, "step": STEP,
                     "fit_on": "검증 2023~24 + 테스트 2025-01~2026-09 (학습에 안 쓴 LightGBM 예측, 소매, 정답 확정분)"}, M / f"prob_{d}.joblib")
        coef.append({"분야": nm, "평가용 a": round(a, 3), "평가용 b": round(b, 3), "서비스 a": round(float(sa), 3), "서비스 b": round(float(sb), 3),
                     "예측 2배 → 확률 +%p (서비스)": round(float(sb) * np.log(2) * 100, 1), "상한 (서비스)": cap, "상한 (평가용)": cap_e})
        # 그래프: 카드 규칙 그대로 (보합 ±1% / 50% 이하 주의 문구 / 1% 단위 / 분야별 상한) + 학습 밖 예측의 실제 적중률
        X = 60
        x = np.linspace(0.01, X / 100, 400)
        ax.axvspan(0, 1, color="#eee", lw=0)
        ax.axhspan(0, 50, color="#f6f6f6", lw=0)
        raw = prob(sa, sb, x)
        ax.plot(x * 100, raw * 100, color="#d1495b", lw=1, ls=":", zorder=4, label="상한이 없을 때")
        ax.plot(x * 100, np.minimum(raw, cap) * 100, color="#d1495b", lw=2.2, zorder=5, label="카드 확률 (상한 = 예측 상위 1%의 실제 적중률)")
        sx, sy = np.r_[sv, st], np.r_[hv, ht]
        o = np.argsort(sx); sx, sy = sx[o], sy[o]
        grp = np.minimum(np.arange(len(sx)) // PER, len(sx) // PER - 1)    # 1,000행씩 (마지막 묶음은 나머지 포함)
        ax.scatter([np.median(sx[grp == k]) * 100 for k in range(grp.max() + 1)], [sy[grp == k].mean() * 100 for k in range(grp.max() + 1)], s=6, color="#2e86ab", zorder=3,
                   label=f"실제 적중률 (2023~2026-09 학습 밖 예측, {PER:,}행씩 {grp.max() + 1}개 전부)")
        ax.set_xlim(0, X); ax.set_ylim(20, 102)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_title(f"{nm}   확률 = {sa:.3f} + {sb:.3f}·ln|예측|   (예측 2배 → +{sb * np.log(2) * 100:.0f}%p)", fontsize=9.5)
        ax.set_xlabel("예측 변화 크기 (절댓값)")
    axes[0].set_ylabel("예측 방향으로 1% 넘게 움직일 확률 (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(F / "direction_prob_curve.png", dpi=150)
    pd.DataFrame(check).to_csv(T / "direction_prob_check.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("■ 계수 (확률 = a + b·ln|예측 크기|)\n", pd.DataFrame(coef).to_string(index=False))
    print("\n■ 평가: 선형 vs 로지스틱 vs 등위 회귀 (검증으로 맞춤, 테스트 Brier)\n", pd.DataFrame(cmp_).to_string(index=False))
    print("\n■ 평가: 검증으로 맞춘 식의 테스트 확인 (표시 → 실제)\n", pd.DataFrame(check).to_string(index=False))
    print("\n그래프 저장: eda/figures/direction_prob_curve.png")