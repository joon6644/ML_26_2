# [현재 결과] 지금 보고서·서비스 숫자를 만드는 스크립트 (eda/README.md)
# 비즈니스 KPI: 카드를 따라 사면 얼마나 아끼나, 틀리면 얼마나 손해인가 (테스트 2025-01~2026-09, 소매, 서비스 모델 예측)
#   선택: 4주치 식재료를 '지금 한 번에'(가격 = 최근 7일 평균) 또는 '4주에 나눠'(가격 = 향후 4주 평균 = 기준가 × (1 + y)) 산다
#   카드 따르기: 상승 예상(예측 > +1%)이면 지금, 하락 예상(< −1%)이면 나눠서, 보합이면 기준 행동 그대로
import sys
from pathlib import Path
import joblib
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
from shared import NAMES, P, direction_prob, load_prob   # noqa: E402
T = ROOT / "eda" / "tables"
rows, loss = [], []
for d, nm in NAMES.items():
    t = pd.read_parquet(P / d / "model_input.parquet", columns=["split", "se_nm", "y", "base"])
    m = (t.split == "테스트").values
    t = t[m].copy(); t["p"] = np.load(T / f"final_test_preds_{d}.npz")["lgb"]
    t = t[(t.se_nm == "소매") & t.y.notna()]
    y, p, base = t.y.values.astype(float), t.p.values, t.base.values.astype(float)
    up, dn = p > 0.01, p < -0.01
    prob = direction_prob(load_prob(d), p)
    now, spread = np.ones(len(y)), 1 + y                     # 상대 비용 (기준가 = 1)
    best = np.minimum(now, spread)
    for name, base_cost in [("항상 지금 사기", now), ("항상 나눠 사기", spread)]:
        card = np.where(up, now, np.where(dn, spread, base_cost))
        save = base_cost - card                                # +면 절약
        room = base_cost - best                                # 미래를 알 때 최대 절약
        rows.append({"분야": nm, "비교 대상": name, "평균 절약 (%)": save.mean() * 100, "최대 가능 절약 (%)": room.mean() * 100,
                     "최대 대비 (%)": save.sum() / room.sum() * 100, "평균 절약 (원/kg)": (save * base).mean()})
    # 틀린 예측의 손실: 카드를 따랐는데 반대로 움직임
    for lab, mm, extra in [("상승 예상 → 지금 샀는데 내림", up & (y < -0.01), -y), ("하락 예상 → 미뤘는데 오름", dn & (y > 0.01), y)]:
        sel = mm
        loss.append({"분야": nm, "틀린 경우": lab, "방향 예측 중 비율 (%)": sel.sum() / (up | dn).sum() * 100,
                     "평균 손해 (%)": extra[sel].mean() * 100, "평균 손해 (원/kg)": (extra * base)[sel].mean(),
                     "손해 10% 넘은 비율 (%)": (extra[sel] > 0.10).mean() * 100})
    # 확률이 높은 카드만 따를 때
    for th in (0.5, 0.6, 0.7):
        follow = (up | dn) & (prob > th)
        card = np.where(follow & up, now, np.where(follow & dn, spread, spread))
        save = spread - card
        rows.append({"분야": nm, "비교 대상": f"항상 나눠 사기 · 확률 {th:.0%} 넘는 카드만 따름", "평균 절약 (%)": save.mean() * 100,
                     "최대 가능 절약 (%)": (spread - best).mean() * 100, "최대 대비 (%)": save.sum() / (spread - best).sum() * 100,
                     "평균 절약 (원/kg)": (save * base).mean()})
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
k = pd.DataFrame(rows); l = pd.DataFrame(loss)
k.to_csv(T / "business_kpi.csv", index=False, encoding="utf-8-sig"); l.to_csv(T / "business_loss.csv", index=False, encoding="utf-8-sig")
print(k.to_string(index=False)); print(); print(l.to_string(index=False))
