# [현재 결과] 지금 보고서·서비스 숫자를 만드는 스크립트 (eda/README.md)
# 1인 가구 1년 장보기 시나리오 (2024년, 검증 기간, 전국 소매가, 1차 모델 RandomForest 예측) → eda/tables/report2_shopping_2024.csv
#   주 1회 장보기, 매주 1주일치를 먹는다. 품목마다 1년 합계 100만 원이 되도록 1주 양을 맞춤 (모델 없이 기준).
#   재고(몇 주치 남았나)를 따라가며 계산 — '늦게 사기'는 집에 남은 게 있을 때만 가능하므로.
#   모델 없이: 재고를 2주치(이번 주 먹을 것 + 여유 1주)로 유지 = 매주 1주일치를 그 주 가격(최근 7일 평균)에 산다.
#   미리 사기만: 상승 예상(> +1%)이면 4주치까지 채우고, 그 외엔 평소처럼 2주치로.
#   미리 사기 + 늦게 사기: 위에 더해 하락 예상(< −1%)이면 여유분을 쓰며 1주치(이번 주 먹을 것)까지만 채움 → 싸진 뒤에 채움.
#   시작 재고 2주치는 두 경우 같고, 연말에 남은 재고 차이는 연말 가격으로 쳐서 뺀다 (같은 양을 먹은 것으로 맞춤).
from pathlib import Path
import numpy as np, pandas as pd
T = Path(__file__).resolve().parent / "tables"
CASES = [("agri", "농산물", "배", "신고", "상품"), ("fishery", "수산물", "고등어", "국산(염장)", "-"), ("livestock", "축산물", "닭", "육계11호", "육계11호"),
         ("agri", "농산물", "양파", "양파", "상품"), ("livestock", "축산물", "돼지", "삼겹살", "삼겹살")]       # 4주 보관 가능한 대중 품목
BASKET = [("agri", "쌀", "10kg", "상품"), ("livestock", "계란", "특란30구", "일반란"), ("agri", "양파", "양파", "상품"), ("agri", "파", "대파", "상품"),
          ("agri", "감자", "수미(노지)", "상품"), ("agri", "당근", "무세척(국산)", "상품"), ("agri", "깐마늘(국산)", "깐마늘(국산)", "상품"),
          ("livestock", "돼지", "앞다리", "앞다리"), ("livestock", "닭", "육계(kg)", "육계(kg)"), ("fishery", "김", "구운김", "-")]   # 자취생 장바구니: 자주 사고 4주 보관 가능한 것 (대파는 썰어 냉동), 보통 사는 형태로 (우유·애호박은 보관이 안 돼 제외)
BUDGET, NORMAL, UP, DOWN, BAND = 1_000_000, 2, 4, 1, 0.01


def simulate(w, kg, delay):
    """주마다: 목표 재고까지 사고 → 1주치 먹음. 반환: 총 지출, 연말 재고(주)."""
    stock, cost = NORMAL, 0.0
    for _, x in w.iterrows():
        target = UP if x.p > BAND else (DOWN if delay and x.p < -BAND else NORMAL)
        buy = max(0, target - stock)
        cost += buy * x.base * kg; stock += buy
        stock -= 1                                                    # 이번 주 먹음 (재고는 항상 1주치 이상이라 굶지 않음)
    return cost, stock


rows = []
for d, nm, item, vrty, grd in CASES:
    r = pd.read_parquet(T / f"report2_rf_val_preds_{d}.parquet")
    r = r[(r.item_nm == item) & (r.vrty_nm == vrty) & (r.grd_nm == grd) & (r.sgg_nm == "전국") & (r.date.dt.year == 2024)].sort_values("date")
    r["week"] = r.date.dt.isocalendar().week.values
    w = r.groupby("week").first().reset_index()                       # 그 주 첫 조사일에 장을 본다
    kg = BUDGET / simulate(w.assign(p=0.0), 1.0, False)[0]            # 모델 없이 1년 지출이 정확히 100만 원이 되는 1주 양
    plain, plain_end = simulate(w.assign(p=0.0), kg, False)           # 모델 없이 = 예측이 늘 보합
    last = w.base.iloc[-1] * kg
    res = {}
    for lab, delay in [("미리 사기만", False), ("미리 + 늦게", True)]:
        cost, end = simulate(w, kg, delay)
        res[lab] = cost - (end - plain_end) * last                    # 연말 재고 차이는 연말 가격으로 정산
    rows.append({"분야": nm, "품목": f"{item} ({vrty} {grd})", "장 본 주": len(w), "모델 없이 (원)": round(plain),
                 "미리 사기만 (원)": round(res["미리 사기만"]), "미리 사기만 절약 (%)": round((plain - res["미리 사기만"]) / plain * 100, 1),
                 "미리 + 늦게 (원)": round(res["미리 + 늦게"]), "미리 + 늦게 절약 (원)": round(plain - res["미리 + 늦게"]),
                 "미리 + 늦게 절약 (%)": round((plain - res["미리 + 늦게"]) / plain * 100, 1),
                 "상승 예상 주": int((w.p > BAND).sum()), "하락 예상 주": int((w.p < -BAND).sum()),
                 "1년 가격 폭 (최대/최소)": round(w.base.max() / w.base.min(), 2)})
out = pd.DataFrame(rows); out.to_csv(T / "report2_shopping_2024.csv", index=False, encoding="utf-8-sig")
# 자취생 장바구니: 같은 규칙, 품목별 절약률 — 검증 기간 두 해(2023, 2024)를 따로 계산해 한 해의 우연인지 확인
bk = []
for d, item, vrty, grd in BASKET:
    row = {"품목": f"{item} ({vrty} {grd})"}
    for yr in (2023, 2024):
        r = pd.read_parquet(T / f"report2_rf_val_preds_{d}.parquet")
        r = r[(r.item_nm == item) & (r.vrty_nm == vrty) & (r.grd_nm == grd) & (r.sgg_nm == "전국") & (r.date.dt.year == yr)].sort_values("date")
        r["week"] = r.date.dt.isocalendar().week.values
        w = r.groupby("week").first().reset_index()
        kg = BUDGET / len(BASKET) / simulate(w.assign(p=0.0), 1.0, False)[0]   # 품목당 1년 10만 원
        plain, pe = simulate(w.assign(p=0.0), kg, False)
        c, e = simulate(w, kg, True); c -= (e - pe) * w.base.iloc[-1] * kg
        row[f"{yr} 가격 폭 (최고/최저)"] = round(w.base.max() / w.base.min(), 1)
        row[f"{yr} 절약 (%)"] = round((plain - c) / plain * 100, 1)
    bk.append(row)
bk = pd.DataFrame(bk)
bk.loc[len(bk)] = {"품목": "장바구니 전체 (품목당 같은 금액)", "2023 절약 (%)": round(bk["2023 절약 (%)"].mean(), 1), "2024 절약 (%)": round(bk["2024 절약 (%)"].mean(), 1)}
bk.to_csv(T / "report2_basket.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 240); print(out.to_string(index=False)); print(); print(bk.to_string(index=False))
