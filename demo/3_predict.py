"""3단계: 저장된 모델로 예측 시연 (소매가 기준)

    python demo/3_predict.py 양파 서울 2025-08-01     # 품목·지역·날짜 하나 예측 → 전망 카드 + 그래프 + 실제 결과
    python demo/3_predict.py 고등어 부산              # 날짜 생략 → 테스트 기간 중 무작위 날짜
    python demo/3_predict.py --perf                   # 테스트 기간(2025-01~) 전체 성능표 (분야별 베이스라인 vs 모델)

먼저 python demo/0_train_models.py 로 모델을 저장해 둘 것. 테스트 기간(2025-01 ~ 2026-09)은 학습에 쓰지 않은 기간이라
'모델이 처음 보는 날짜'에 대한 예측과 실제 결과를 나란히 볼 수 있다.
"""
import sys

import numpy as np
import pandas as pd

from shared import NAMES, P, ROOT, TEST_START, BAND, SIZE_BINS, direction_prob, label, load_models, load_prob, metrics, predict, prob_text, size_bin

OUT = ROOT / "demo" / "output"
ARROW = {1: "상승 예상", -1: "하락 예상", 0: "보합 (±1% 이내)"}


def domain_of(item):
    m = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str)
    hit = m[m.item_nm == item]
    if hit.empty:
        sys.exit(f"'{item}' 품목이 없습니다. 예: " + ", ".join(m.item_nm.sample(12, random_state=1)))
    return hit.domain.iloc[0]


HIST = ["r7", "r28", "r91", "yoy", "ly_change"]   # 가격 이력 피처 — 4개 이상 비면 '조사 이력이 짧음' 경고


def card(item, region, date=None):
    """표시 규칙: 지역 전망 → (지역 데이터 부족) 전국 전망 + 안내 → (둘 다 부족) 전망 없음 + 마지막 조사 가격.
    전망을 내는 행이라도 가격 이력 피처가 대부분 비어 있으면 정확도 경고를 붙인다."""
    d = domain_of(item)
    df = pd.read_parquet(P / d / "model_input.parquet", filters=[("item_nm", "==", item), ("se_nm", "==", "소매")])
    if df.empty:
        sys.exit(f"'{item}' 품목은 소매가가 조사되지 않아 전망을 제공하지 않습니다 (중도매만 조사).")
    reg, nat = df[df.sgg_nm == region], df[df.sgg_nm == "전국"]
    if date is None:
        pool = reg if len(reg) else nat
        date = pool[pool.date >= TEST_START].date.sample(1).iloc[0]
    date = pd.Timestamp(date)
    note = ""
    rows, s = reg[reg.date == date], reg
    if rows.empty:                               # 그날 지역 전망이 없으면 (최근 7일 조사 3일 미만 등) 전국으로 대체
        rows, s = nat[nat.date == date], nat
        note = f"{region}은 최근 조사가 부족해 전국 평균 가격으로 대신 보여드립니다"
    print(f"\n[{NAMES[d]}] {item} · {region} · 기준일 {date:%Y-%m-%d}" + (" (학습에 안 쓴 기간)" if date >= pd.Timestamp(TEST_START) else " (학습 기간)"))
    if rows.empty:                               # 전국도 없으면 전망 없음
        last = df[df.date <= date].sort_values("date").tail(1)
        msg = f" — 마지막 조사 {last.date.iloc[0]:%Y-%m-%d} ({last.sgg_nm.iloc[0]}) 최근 7일 평균 {float(last.base.iloc[0]):,.0f} 원/kg" if len(last) else ""
        print(f"\n  최근 조사 없음 · 전망을 제공하지 않습니다{msg}")
        return
    if note:
        print(f"  ※ {note}")
    p = predict(load_models(d), rows)
    prob = direction_prob(load_prob(d), p)
    for i, (_, r) in enumerate(rows.iterrows()):
        base, y = float(r.base), float(r.y)
        print(f"\n  {r.vrty_nm} · {r.grd_nm}")
        print(f"  최근 7일 평균        {base:>10,.0f} 원/kg")
        print(f"  ▶ 향후 4주 평균 전망  {base * (1 + p[i]):>10,.0f} 원/kg  ({p[i]:+.1%}, {ARROW[int(label(p[i:i + 1])[0])]})")
        print(f"                       {prob_text(prob[i], p[i])}")
        if r[HIST].isna().sum() >= 4:
            print("                       ※ 조사 이력이 짧아 정확도가 낮을 수 있습니다")
        if abs(p[i]) <= BAND:
            hit = "보합 예측이라 방향 판정 안 함" + (" (실제로도 ±1% 이내)" if abs(y) <= BAND else "")
        else:
            hit = "방향 적중" if abs(y) > BAND and np.sign(p[i]) == np.sign(y) else "방향 틀림" if abs(y) > BAND else "실제 변화 ±1% 이내"
        print(f"  ▷ 실제 4주 평균      {base * (1 + y):>10,.0f} 원/kg  ({y:+.1%}) → {hit}")
    plot(s, rows.iloc[0], p[0], item, region, date)


def plot(s, r, p, item, region, day):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    g = s[(s.vrty_nm == r.vrty_nm) & (s.grd_nm == r.grd_nm)].set_index("date").base
    g = g[(g.index >= day - pd.Timedelta(days=365)) & (g.index <= day + pd.Timedelta(days=60))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g[g.index <= day], color="#555", lw=1.5, label="최근 7일 평균 가격 (기준일까지)")
    ax.plot(g[g.index > day], color="#bbb", lw=1.5, label="이후 실제 가격 (모델은 모름)")
    end = day + pd.Timedelta(days=28)
    ax.hlines(r.base * (1 + p), day, end, color="#d1495b", lw=3, label=f"모델 전망: 4주 평균 {p:+.1%}")
    ax.hlines(r.base * (1 + r.y), day, end, color="#2e86ab", lw=3, ls="--", label=f"실제 4주 평균 {r.y:+.1%}")
    ax.axvline(day, color="#999", lw=0.8, ls=":")
    ax.set_title(f"{item} · {region} · {r.vrty_nm} {r.grd_nm} (소매, 원/kg)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    f = OUT / f"predict_{item}_{region}_{day:%Y%m%d}.png"
    fig.savefig(f, dpi=150)
    print(f"\n  그래프 저장: {f.relative_to(ROOT)}")
    plt.show()


def perf():
    rows, sizes, probs = [], [], []
    for d in ["agri", "livestock", "fishery"]:
        df = pd.read_parquet(P / d / "model_input.parquet", filters=[("se_nm", "==", "소매"), ("split", "==", "테스트")])
        df = df[df.y.notna()]
        y, p = df.y.values.astype(float), predict(load_models(d), df)
        for nm, pp in [("베이스라인", np.zeros(len(y))), ("LightGBM", p)]:
            rows.append({"분야": NAMES[d], "모델": nm, **{k: round(v, 4) for k, v in metrics(y, pp).items()}})
        ok, mv, b = np.sign(p) == np.sign(y), np.abs(y) > BAND, size_bin(p)
        for name in [n for _, _, n in SIZE_BINS]:
            m = b == name
            sizes.append({"분야": NAMES[d], "예측 크기": name, "비율": f"{m.mean():.0%}", "방향 정확도": round(float(ok[m & mv].mean()), 3)})
        pr, real = direction_prob(load_prob(d), p), np.sign(p) * y > BAND
        for lab, m in [("50% 이하 → 주의 문구", pr <= 0.5)] + [(f"{x - 0.04:.0%}~{x:.0%}", (pr > x - 0.05 + 1e-9) & (pr <= x + 1e-9)) for x in np.arange(0.55, 0.851, 0.05)]:   # 5%p 구간으로 묶어 확인
            if m.any():
                probs.append({"분야": NAMES[d], "표시": lab, "실제": f"{real[m].mean():.0%}"})
        print(f"  {NAMES[d]} 예측 {len(df):,}행 완료", flush=True)
    pd.set_option("display.width", 200)
    print("\n테스트 성능 (2025-01 ~ 2026-09, 소매, 학습에 안 쓴 기간)\n방향 정확도 = 실제 변화가 ±1%를 넘은 경우 오름·내림 적중률 / 베이스라인 = 최근 7일 평균 유지(항상 0%)\n")
    print(pd.DataFrame(rows).fillna("–").to_string(index=False))
    print("\n예측 크기별 방향 정확도 (예측 변화가 클수록 방향이 잘 맞는다)\n")
    print(pd.DataFrame(sizes).pivot(index="예측 크기", columns="분야", values="방향 정확도").reindex([n for _, _, n in SIZE_BINS])[list(NAMES.values())].to_string())
    print("\n카드에 표시하는 확률이 실제로 맞나 (표시 확률 → 실제로 그 방향으로 1% 넘게 움직인 비율)\n")
    print(pd.DataFrame(probs).pivot(index="표시", columns="분야", values="실제")[list(NAMES.values())].fillna("–").to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--perf":
        perf()
    else:
        card(a[0], a[1] if len(a) > 1 else "서울", a[2] if len(a) > 2 else None)
