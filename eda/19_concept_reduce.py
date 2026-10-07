# %% [markdown]
# # 같은 개념 피처 묶음에서 대표 1개만 남겨도 성능이 유지되는가 (전국 모델, LightGBM, 시드 3개)
# 묶음은 상관이 아니라 '같은 개념'으로 정의 (기온끼리, 유가끼리, 같은 가격의 다른 출처끼리 …)
# 대표 = 묶음 안 SHAP 비중(eda/17) 최대. 나머지는 모두 동시에 제거. 학습 2016~2022 / 테스트 2023~2024
# 실행: `python eda/19_concept_reduce.py` → eda/19_concept_reduce.md

# %%
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("fd", ROOT / "eda" / "17_feature_decision.py")
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)
SEEDS = (0, 1, 2)

CONCEPT = {
    "기온": ["wx_temp_avg", "wx_temp_max", "wx_temp_min", "wx_ground_temp", "prodarea_temp_avg", "prodarea_temp_max", "prodarea_temp_min"],
    "풍속": ["wx_wind_avg", "wx_wind_gust", "prodarea_wind_avg"],
    "강수": ["wx_rain", "wx_rain_1h_max", "prodarea_rain"],
    "일조·일사": ["wx_sunshine_hr", "wx_solar_rad", "prodarea_sunshine_hr"],
    "습도": ["wx_humidity", "prodarea_humidity"],
    "유가": ["intl_원유_Brent", "fuel_보통휘발유", "fuel_자동차용_경유", "fuel_실내등유"],
    "소비자물가": ["cpi_총지수", "cpi_식료품"],
    "수입": ["import_kg", "import_usd"],
    "도매가 (출처만 다름)": ["whsl_price_kg", "mafra_whsl_price_kg"],
    "가락 거래 규모": ["garak_qty_kg", "garak_amount_krw"],
    "수입물가: 과일·채소": ["impidx_과일", "impidx_채소및과실"],
    "수입물가: 곡물": ["impidx_곡물및식량작물", "impidx_맥류및잡곡", "impidx_곡류"],
    "수입물가: 축산 전체": ["impidx_축산물", "impidx_낙농및육류", "impidx_도축육"],
    "수입물가: 낙농": ["impidx_낙농품", "impidx_육가공품및낙농품"],
    "수입물가: 냉동수산": ["impidx_냉동수산물", "impidx_냉동건조수산물"],
    "수입물가: 수산가공": ["impidx_수산가공품", "impidx_수산물가공품"],
    "돼지 경락가 (탕박·대표가)": ["pig_auction_price_kg", "pig_auction_rep_kg"],
    "금어기": ["closed_season", "closed_window"],
    "적조": ["redtide_reports", "redtide_areas"],
    "해수온": ["sea_temp_east", "sea_temp_south", "sea_temp_west", "buoy_temp_max", "coast_temp_avg"],
    "어장 수온 (표층·저층)": ["farm_temp_s", "farm_temp_b"],
    "품목 어장 수온 (표층·저층)": ["farmitem_temp_s", "farmitem_temp_b"],
    "어장 용존산소 (표층·저층)": ["farm_do_s", "farm_do_b"],
    "품목 어장 용존산소 (표층·저층)": ["farmitem_do_s", "farmitem_do_b"],
}


def runs(f, num, tr, te):
    r = [fd.fit(f, num, tr, te, seed=s) for s in SEEDS]
    return np.mean([x["RMSE"] for x in r]), np.std([x["RMSE"] for x in r]), np.mean([x["R2"] for x in r])


def main():
    t_all = time.time()
    L = ["# 같은 개념 피처 → 대표 1개만 남기기", "", "> 자동 생성: `python eda/19_concept_reduce.py` · LightGBM 시드 3개 평균, 학습 2016~2022 / 테스트 2023~2024",
         "> 대표 = 묶음 안 SHAP 비중 최대. 판정: |ΔRMSE| ≤ 2 × √(전체 std² + 축소 std²) 이면 유지", ""]
    S = []
    for d, nm in fd.NAMES.items():
        f = pd.read_parquet(fd.P / d / "features.parquet")
        t0, t1, s0, s1 = fd.SPLIT
        tr = ((f.date >= t0) & (f.date <= t1)).values
        te = ((f.date >= s0) & (f.date <= s1)).values
        num = [c for c in f.columns if c not in fd.IDS]
        shap = pd.read_csv(fd.TAB / f"feature_decision_{d}.csv").set_index("피처")["SHAP %"]
        drop, rows = [], []
        for g, cols in CONCEPT.items():
            cols = [c for c in cols if c in num]
            if len(cols) < 2:
                continue
            rep = max(cols, key=lambda c: shap.get(c, 0))
            drop += [c for c in cols if c != rep]
            rows.append({"개념": g, "대표 (남김)": f"{rep} ({shap.get(rep, 0):.2f}%)",
                         "제거": ", ".join(f"{c} ({shap.get(c, 0):.2f}%)" for c in cols if c != rep)})
        full = runs(f, num, tr, te)
        red = runs(f, [c for c in num if c not in drop], tr, te)
        diff, noise = red[0] - full[0], np.hypot(full[1], red[1])
        verdict = "유지 (시드 잡음 이내)" if abs(diff) <= 2 * noise else ("나빠짐" if diff > 0 else "좋아짐")
        S += [{"분야": nm, "피처 세트": f"전체 {len(num) + 3}개", "RMSE": round(full[0], 4), "± (시드)": round(full[1], 5), "R²": round(full[2], 3)},
              {"분야": nm, "피처 세트": f"개념 축소 {len(num) + 3 - len(drop)}개 (−{len(drop)})", "RMSE": round(red[0], 4),
               "± (시드)": round(red[1], 5), "R²": round(red[2], 3), "ΔRMSE": round(diff, 5), "판정": verdict}]
        L += [f"## {nm}: 묶음 {len(rows)}개, {len(drop)}개 제거 → ΔRMSE {diff:+.5f} (기준 ±{2 * noise:.5f}) → **{verdict}**", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]
        print(f"{nm}: −{len(drop)}개, 전체 {full[0]:.4f}±{full[1]:.5f} → 축소 {red[0]:.4f}±{red[1]:.5f}, ΔRMSE {diff:+.5f} → {verdict}", flush=True)
    S = pd.DataFrame(S)
    L = L[:4] + ["## 요약", "", S.fillna("").to_markdown(index=False), ""] + L[4:] + [f"> 전체 소요 {time.time() - t_all:.0f}초"]
    (ROOT / "eda" / "19_concept_reduce.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
