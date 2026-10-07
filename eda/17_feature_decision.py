# %% [markdown]
# # 피처 정리 의사결정 자료 (전국 모델, features.parquet, 학습 2016~2022 / 테스트 2023~2024)
# A. 피처별: 결측률(학습 기간), 데이터 시작일, y와 Spearman ρ, SHAP 비중(LightGBM, 테스트 2만 행), 가장 비슷한 피처와 ρ
# B. 묶음 제거 시험: 후보 묶음을 빼고 LightGBM 재학습 → ΔRMSE, ΔR². 잡음 폭 = 시드 3개 기준 모델 RMSE 표준편차
# 실행: `python eda/17_feature_decision.py` → eda/17_feature_decision.md, eda/tables/feature_decision_*.csv

# %%
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_features as bf  # noqa: E402

P = ROOT / "data" / "processed"
TAB = ROOT / "eda" / "tables"
SPLIT = ("2016-01-01", "2022-12-01", "2023-01-01", "2024-12-31")
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
IDS = bf.ID + ["base", "y"]
FUEL = ["fuel_보통휘발유", "fuel_자동차용_경유", "fuel_실내등유"]
ECON_ALL = ["usd_krw", "cpi_총지수", "cpi_식료품", "cpi_item", "intl_원유_Brent", "intl_천연가스", "intl_소맥", "intl_옥수수", "intl_대두"] + FUEL
GROUP = {
    "공통": {
        "지면온도 제거 (평균기온과 0.99)": ["wx_ground_temp"],
        "최고·최저기온 제거 (평균기온만)": ["wx_temp_max", "wx_temp_min"],
        "휘발유·등유 제거 (경유만)": ["fuel_보통휘발유", "fuel_실내등유"],
        "유가 3종 제거 (브렌트유만)": FUEL,
        "물가 총지수 제거 (식료품만)": ["cpi_총지수"],
        "최대순간풍속 제거": ["wx_wind_gust"],
        "1시간 최대강수 제거": ["wx_rain_1h_max"],
        "일사량 제거 (일조시간과 0.81)": ["wx_solar_rad"],
        "수입액 제거 (수입량과 0.88~0.97)": ["import_usd"],
        "요일 제거": ["dow"],
        "[참고] 전국 기상 전체 제거": bf.WX,
        "[참고] 경제 지표 전체 제거": ECON_ALL,
        "[참고] 수입물가지수 전체 제거": "impidx_*",
    },
    "agri": {
        "주산지 기상 7개 제거 (전국 기상과 0.95~1.00)": "prodarea_*",
        "MAFRA 도매가 제거 (aT 중도매가와 0.98)": ["mafra_whsl_price_kg"],
        "가락 정산금액 제거": ["garak_amount_krw"],
        "수입물가 중복 6개 제거": ["impidx_곡류", "impidx_곡물및식량작물", "impidx_맥류및잡곡", "impidx_과일", "impidx_콩류", "impidx_유지"],
        "[참고] 가락시장 5개 전체 제거": "garak_*",
    },
    "livestock": {
        "품목 소비자물가 제거 (결측 97%)": ["cpi_item"],
        "돼지 대표가 제거 (탕박 경락가와 1.00)": ["pig_auction_rep_kg"],
        "수입물가 중복 3개 제거": ["impidx_낙농및육류", "impidx_도축육", "impidx_육가공품및낙농품"],
        "[참고] 가축질병 4개 전체 제거": "disease_*",
        "[참고] 사육·경락·재고 전체 제거": ["census_heads", "census_farms", "auction_heads", "auction_price_kg", "stock_ton",
                                     "pig_auction_price_kg", "pig_auction_skinned_kg", "pig_auction_rep_kg"],
    },
    "fishery": {
        "MAFRA 도매가 제거 (aT 중도매가와 0.99)": ["mafra_whsl_price_kg"],
        "수입물가 중복 2개 제거": ["impidx_냉동건조수산물", "impidx_수산물가공품"],
        "금어기 여부 제거 (가능 기간과 1.00)": ["closed_season"],
        "적조 해역 수 제거 (보고 수와 0.98)": ["redtide_areas"],
        "부이최고·연안수온·관측소수 제거": ["buoy_temp_max", "coast_temp_avg", "coast_temp_n_stations"],
        "어장 저층수온 2개 제거": ["farm_temp_b", "farmitem_temp_b"],
        "[참고] 품목 어장환경 10개 전체 제거": "farmitem_*",
        "[참고] 전체 어장환경 10개 전체 제거": "farm_*",
        "[참고] 적조·해파리 4개 전체 제거": ["redtide_reports", "redtide_areas", "redtide_density_max", "jellyfish_reports"],
    },
}


def expand(spec, cols):
    if isinstance(spec, str):
        pre = spec.rstrip("*")
        return [c for c in cols if c.startswith(pre) and not (pre == "farm_" and c.startswith("farmitem_"))]
    return [c for c in spec if c in cols]


def fit(f, num, tr, te, seed=0, shap=False):
    X = f[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr].mean()).fillna(0)
    Xc = pd.DataFrame(MinMaxScaler().fit(X[tr]).transform(X), columns=num, index=f.index).astype("float32")
    for c in bf.CAT:
        Xc[c] = f[c].astype(str).astype(pd.CategoricalDtype(sorted(f[c].astype(str).unique())))
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(Xc[tr], f.y[tr], categorical_feature=bf.CAT)
    p, y = m.predict(Xc[te]), f.y[te].values
    out = {"RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}
    if shap:
        idx = np.random.default_rng(0).choice(np.where(te)[0], min(20000, te.sum()), replace=False)
        sv = np.abs(m.predict(Xc.iloc[idx], pred_contrib=True)[:, :-1]).mean(0)
        out["shap"] = pd.Series(sv / sv.sum() * 100, index=Xc.columns)
    return out


def main():
    t_all = time.time()
    L = ["# 피처 정리 의사결정 자료", "", "> 자동 생성: `python eda/17_feature_decision.py` · 전국 모델(features.parquet), LightGBM, 학습 2016~2022 / 테스트 2023~2024",
         "> ρ(y) = 학습 기간 Spearman. SHAP % = 테스트 2만 행 평균 |SHAP| 비중. 결측 % = 학습 기간. 비슷한 피처 = 같은 분야 피처 중 |ρ| 최대",
         "> 제거 시험: ΔRMSE > 0 이면 빼서 나빠짐(그 변수가 도움이 됨), < 0 이면 빼서 좋아짐. 잡음 폭(시드 3개 RMSE 표준편차)보다 작으면 차이 없음으로 본다", ""]
    for d, nm in NAMES.items():
        t = time.time()
        f = pd.read_parquet(P / d / "features.parquet")
        t0, t1, s0, s1 = SPLIT
        tr = ((f.date >= t0) & (f.date <= t1)).values
        te = ((f.date >= s0) & (f.date <= s1)).values
        num = [c for c in f.columns if c not in IDS]
        base = [fit(f, num, tr, te, seed=s, shap=(s == 0)) for s in (0, 1, 2)]
        rm, rs = np.mean([b["RMSE"] for b in base]), np.std([b["RMSE"] for b in base])
        r2 = np.mean([b["R2"] for b in base])
        shap = base[0]["shap"]
        # A. 피처별 표
        ftr = f[tr]
        smp = ftr.sample(min(200000, len(ftr)), random_state=0)
        corr = smp[num].corr(method="spearman")
        rows = []
        for c in num + bf.CAT:
            nn = ftr[c].notna()
            first = f.loc[f[c].notna(), "date"].min()
            if c in num:
                cc = corr[c].drop(c).abs()
                mate, mv = (cc.idxmax(), corr.loc[c, cc.idxmax()]) if cc.notna().any() else ("", np.nan)
                ry = smp[c].corr(smp.y, method="spearman")
            else:
                mate, mv, ry = "", np.nan, np.nan
            rows.append({"피처": c, "구분": "공통" if c in bf.COMMON + bf.HIST + ["woy"] + bf.CAT else "분야 전용",
                         "결측 %": round((1 - nn.mean()) * 100, 1), "시작": f"{first:%Y-%m}" if pd.notna(first) else "-",
                         "ρ(y)": round(ry, 3) if pd.notna(ry) else "", "SHAP %": round(shap.get(c, np.nan), 2),
                         "가장 비슷한 피처": mate, "그 ρ": round(mv, 2) if pd.notna(mv) else ""})
        A = pd.DataFrame(rows).sort_values(["구분", "SHAP %"], ascending=[True, False])
        A.to_csv(TAB / f"feature_decision_{d}.csv", encoding="utf-8-sig", index=False)
        # B. 묶음 제거 시험
        groups = {**GROUP["공통"], **GROUP[d]}
        B = []
        for g, spec in groups.items():
            drop = expand(spec, num)
            if not drop:
                continue
            o = fit(f, [c for c in num if c not in drop], tr, te)
            dr = o["RMSE"] - base[0]["RMSE"]
            B.append({"제거 묶음": g, "변수 수": len(drop), "ΔRMSE": round(dr, 5), "ΔR²": round(o["R2"] - base[0]["R2"], 4),
                      "판정": "차이 없음 (잡음 이내)" if abs(dr) <= 2 * rs else ("빼면 나빠짐" if dr > 0 else "빼면 좋아짐"),
                      "제거 변수": ", ".join(drop)})
            print(f"  {nm} {g}: ΔRMSE {dr:+.5f}", flush=True)
        B = pd.DataFrame(B)
        B.to_csv(TAB / f"feature_ablation_{d}.csv", encoding="utf-8-sig", index=False)
        L += [f"## {nm}", "",
              f"- 기준 모델 (피처 {len(num) + len(bf.CAT)}개): RMSE {rm:.4f} ± {rs:.5f} (시드 3개), R² {r2:.3f}",
              f"- 판정 기준: |ΔRMSE| ≤ {2 * rs:.5f} (잡음 폭 × 2) 이면 차이 없음", "",
              "### B. 묶음 제거 시험", "", B.to_markdown(index=False), "",
              "### A. 피처별 정보", "", A.to_markdown(index=False), ""]
        print(f"{nm} 완료 {time.time() - t:.0f}초", flush=True)
    L.append(f"> 전체 소요 {time.time() - t_all:.0f}초")
    (ROOT / "eda" / "17_feature_decision.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
