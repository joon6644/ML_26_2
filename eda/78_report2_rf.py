# [현재 결과] 지금 보고서·서비스 숫자를 만드는 스크립트 (eda/README.md)
# 과제2 (1차 예측 모델 보고서): 첫 예측 모델 RandomForest — 검증 기간만 사용 (테스트 기간은 쓰지 않음)
#   학습 2016-01 ~ 2022-12-01 (원래 데이터, 정제·타깃 자르기 없음) → 검증 2023-01 ~ 2024-12, 평가는 소매 행
#   → eda/tables/report2_rf.csv (성능), report2_rf_size.csv (예측 크기별 방향 정확도), report2_rf_kpi.csv (절약·손실), report2_rf_cards.csv (예시 후보)
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
from shared import NAMES, BAND, SIZE_BINS, metrics, size_bin   # noqa: E402
P0, T = ROOT / "data" / "processed", ROOT / "eda" / "tables"
CATS = ["item_cd", "food_group", "se_nm", "sgg_nm"]
perf, sizes, kpi, cards = [], [], [], []
for d, nm in NAMES.items():
    t0 = time.time()
    O = pd.read_parquet(P0 / d / "model_frame_v1.parquet")
    num = json.loads((P0 / d / "model_frame_v1.json").read_text(encoding="utf-8"))["num"]
    cats = [c for c in CATS if c in O.columns and c in json.loads((P0 / d / "model_input.json").read_text(encoding="utf-8")).get("cats", CATS)]
    tr = O[(O.date >= "2016-01-01") & (O.date <= "2022-12-01") & O.y.notna()]
    va = O[(O.date >= "2023-01-01") & (O.date <= "2024-12-31") & O.y.notna()]
    def X(df):
        x = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
        for c in num:                                                      # 대상 밖 품목 = 아주 작은 값 (트리가 따로 나눔)
            s = f"_scope__{c}"
            if s in df:
                x.loc[~df[s].astype(bool).values, c] = -1e9
        for c in cats:                                                     # 범주 → 정수 코드 (학습 구간 기준)
            x[c] = pd.Categorical(df[c].astype(str), categories=sorted(tr[c].astype(str).unique())).codes
        return x
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, max_samples=0.1, n_jobs=-1, random_state=42)
    rf.fit(X(tr), tr.y.values.astype(float))
    r = va[va.se_nm == "소매"].copy()
    r["p"] = rf.predict(X(r))
    r[["date", "item_nm", "vrty_nm", "grd_nm", "sgg_nm", "base", "y", "p"]].to_parquet(T / f"report2_rf_val_preds_{d}.parquet", index=False)   # 1년 장보기 시나리오용 (eda/80)
    y, p = r.y.values.astype(float), r.p.values
    for name, pp in [("베이스라인", np.zeros(len(y))), ("RandomForest", p)]:
        m = metrics(y, pp)
        perf.append({"분야": nm, "모델": name, "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4), "R²": round(m["R²"], 3),
                     "방향 정확도": round(m["방향 정확도"], 3) if name != "베이스라인" else None, "학습 행": len(tr), "검증 소매 행": len(r)})
    ok, mv, b = np.sign(p) == np.sign(y), np.abs(y) > BAND, size_bin(p)
    for _, _, s in SIZE_BINS:
        sizes.append({"분야": nm, "예측 크기": s, "방향 정확도": round(float(ok[(b == s) & mv].mean()), 3)})
    # KPI: 평소처럼 4주에 나눠 사는 사람 대비, 카드대로 (상승 예상 → 지금, 하락 예상 → 미룸)
    up, dn = p > BAND, p < -BAND
    spread = 1 + y; card = np.where(up, 1.0, spread); save = spread - card; best = spread - np.minimum(1.0, spread)
    base = r.base.values.astype(float)
    wrong_up, wrong_dn = up & (y < -BAND), dn & (y > BAND)
    loss = np.r_[-y[wrong_up], y[wrong_dn]]
    kpi.append({"분야": nm, "평균 절약 (%)": round(save.mean() * 100, 2), "평균 절약 (원/kg)": round((save * base).mean()), "최대 가능 절약 (%)": round(best.mean() * 100, 2),
                "최대 대비 (%)": round(save.sum() / best.sum() * 100), "반대로 움직인 카드 (%)": round((wrong_up.sum() + wrong_dn.sum()) / (up | dn).sum() * 100, 1),
                "그때 평균 손해 (%)": round(loss.mean() * 100, 1)})
    # 예시 후보: 대표 품목, 서울/부산, 예측이 크고 맞은 경우 + 작게 틀린 경우
    for item, region in [("양파", "서울"), ("배추", "서울"), ("돼지", "서울"), ("고등어", "부산")]:
        c = r[(r.item_nm == item) & (r.sgg_nm == region)]
        if c.empty:
            continue
        c = c.assign(hit=np.sign(c.p) == np.sign(c.y))
        for lab, s in [("큰 예측·적중", c[(c.p.abs() > 0.06) & c.hit & (c.y.abs() > BAND)]), ("작은 예측·틀림", c[(c.p.abs().between(0.012, 0.03)) & ~c.hit & (c.y.abs() > 0.02)])]:
            for _, row in s.sort_values("date").iloc[[len(s) // 2]].iterrows() if len(s) else []:
                cards.append({"분야": nm, "품목": item, "지역": region, "구분": lab, "기준일": row.date.date(), "품종": row.vrty_nm, "등급": row.grd_nm,
                              "최근 7일 평균": round(row.base), "예측": round(row.p, 4), "실제": round(row.y, 4)})
    print(f"{nm} 완료 ({time.time() - t0:.0f}초)", flush=True)
for n, x in [("report2_rf", perf), ("report2_rf_size", sizes), ("report2_rf_kpi", kpi), ("report2_rf_cards", cards)]:
    pd.DataFrame(x).to_csv(T / f"{n}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print(pd.DataFrame(perf).to_string(index=False)); print(pd.DataFrame(sizes).pivot(index="예측 크기", columns="분야", values="방향 정확도").to_string())
print(pd.DataFrame(kpi).to_string(index=False)); print(pd.DataFrame(cards).to_string(index=False))
