"""분야별 학습용 피처 테이블 → data/processed/{agri,livestock,fishery}/features.parquet

    python scripts/build_features.py      # build_dataset.py 다음에 실행

dataset.parquet(원본 변수)에서
- 도메인 기준으로 정리한 변수만 남기고 (공통 38 / 농산 67 / 축산 60 / 수산 82, eda/10_feature_diet.py 참고)
- 가격 이력 8개(r7 … n_obs28)와 주차(woy)를 시계열별 과거 값으로 계산하고
- 타깃 y = 향후 28일 평균 / 최근 7일 평균 − 1 을 붙인다.
결측은 채우지 않고 스케일링도 하지 않는다 (학습셋 평균 대체·MinMax는 모델링 단계에서 학습셋으로 fit).
y가 비어 있는 행(최근 28일처럼 미래 가격이 아직 없는 날, 최근 7일 가격이 부족한 날)은 뺀다 → 학습·평가 전용 테이블.
컬럼 설명: docs/feature_set.md
"""
import numpy as np
import pandas as pd

from common import PROCESSED as P
from common import ROOT

H = 28
KEYS = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
ID = ["domain", "date", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm", "food_group"]
CAT = ["item_cd", "food_group", "se_nm"]                     # 모델 입력 범주형
HIST = ["r7", "r28", "r91", "yoy", "vol28", "dev_last", "ly_change", "n_obs28"]

WX = ["wx_temp_avg", "wx_temp_max", "wx_temp_min", "wx_rain", "wx_rain_1h_max", "wx_snow_depth", "wx_humidity",
      "wx_wind_avg", "wx_wind_gust", "wx_sunshine_hr", "wx_solar_rad", "wx_ground_temp"]
ECON = ["usd_krw", "cpi_총지수", "cpi_식료품", "cpi_item", "intl_원유_Brent", "intl_천연가스",
        "fuel_보통휘발유", "fuel_자동차용_경유", "fuel_실내등유"]
COMMON = ["price_kg", "dow"] + WX + ECON + ["import_kg", "import_usd", "export_kg"]      # + 가격 이력 8 + woy + 범주 3 = 38
DOMAIN = {
    "agri": ["whsl_price_kg", "mafra_whsl_price_kg",
             "garak_qty_kg", "garak_price_kg", "garak_amount_krw", "garak_n_trades", "garak_top_origin_share",
             "prodarea_temp_avg", "prodarea_temp_max", "prodarea_temp_min", "prodarea_rain", "prodarea_sunshine_hr",
             "prodarea_humidity", "prodarea_wind_avg",
             "production_ton", "cultivated_area_ha",
             "intl_소맥", "intl_옥수수", "intl_대두",
             "impidx_농산물", "impidx_곡류", "impidx_곡물및식량작물", "impidx_맥류및잡곡", "impidx_콩류", "impidx_과일",
             "impidx_채소및과실", "impidx_과실및채소가공품", "impidx_기타식용작물", "impidx_유지"],
    "livestock": ["is_holiday",
                  "disease_cases", "disease_heads", "disease_all_cases", "disease_all_heads",
                  "census_heads", "census_farms",
                  "auction_heads", "auction_price_kg", "pig_auction_price_kg", "pig_auction_skinned_kg", "pig_auction_rep_kg",
                  "stock_ton",
                  "intl_소맥", "intl_옥수수", "intl_대두",
                  "impidx_축산물", "impidx_도축육", "impidx_가금육", "impidx_낙농품", "impidx_낙농및육류", "impidx_육가공품및낙농품"],
    "fishery": ["whsl_price_kg", "mafra_whsl_price_kg",
                "catch_ton", "catch_value_kkrw",
                "sea_temp_east", "sea_temp_south", "sea_temp_west", "buoy_temp_max", "buoy_wind_avg", "coast_temp_avg",
                "coast_temp_n_stations",
                "redtide_reports", "redtide_areas", "redtide_density_max", "jellyfish_reports",
                "closed_season", "closed_window", "days_to_closed_start"]
               + [f"{p}_{v}" for p in ("farm", "farmitem") for v in ("temp_s", "temp_b", "sal_s", "do_s", "do_b", "cod_s", "din_s", "dip_s", "chl_s", "ss_s")]
               + ["impidx_수산물", "impidx_신선수산물", "impidx_냉동수산물", "impidx_냉동건조수산물", "impidx_수산가공품", "impidx_수산물가공품"],
}
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}


def rmean(s, lo, hi, minp):
    """t-hi ~ t-lo 구간 평균 (과거 값만)."""
    return s.shift(lo).rolling(hi - lo + 1, min_periods=minp).mean()


def add_history_and_target(df):
    """시계열별 최근 7일 평균(base), 가격 이력 8개, 타깃 y."""
    parts = []
    for _, g in df.groupby(KEYS, observed=True, sort=False):
        s = g.set_index("date").price_kg.asfreq("D")
        base = rmean(s, 0, 6, 3)
        f = pd.DataFrame(index=s.index)
        f["base"] = base
        f["r7"] = np.log(base / rmean(s, 7, 13, 3))
        f["r28"] = np.log(base / rmean(s, 28, 34, 3))
        f["r91"] = np.log(base / rmean(s, 91, 97, 3))
        f["yoy"] = np.log(base / rmean(s, 358, 371, 3))
        f["vol28"] = np.log(s).diff().rolling(28, min_periods=10).std()
        f["dev_last"] = np.log(s / base)
        f["ly_change"] = np.log(s.shift(364 - H).rolling(H, min_periods=10).mean() / rmean(s, 358, 364, 3))
        f["n_obs28"] = s.notna().rolling(28, min_periods=1).sum()
        parts.append(f.reindex(g.date).set_index(g.index))
    out = df.join(pd.concat(parts))
    out[HIST] = out[HIST].replace([np.inf, -np.inf], np.nan)
    out["y"] = out[f"target_mean_{H}d"] / out.base - 1
    return out


DOC = {
    "y": "타깃. 향후 28일(t+1~t+28) 평균 가격 / 최근 7일(t-6~t) 평균 가격 − 1 (0.05 = 5% 상승)",
    "base": "최근 7일 평균 가격 원/kg (모델 입력 아님. 예측 가격 = base × (1 + ŷ))",
    "r7": "최근 7일 평균 / 1주 전 7일 평균 (log)", "r28": "최근 7일 평균 / 4주 전 7일 평균 (log)",
    "r91": "최근 7일 평균 / 13주 전 7일 평균 (log)", "yoy": "최근 7일 평균 / 1년 전 같은 2주 평균 (log)",
    "vol28": "최근 28일 일별 가격 변화율(log)의 표준편차", "dev_last": "오늘 가격 / 최근 7일 평균 (log)",
    "ly_change": "작년 같은 시기의 향후 28일 변화율 (log)", "n_obs28": "최근 28일 중 가격 조사일 수",
    "woy": "ISO 주차 (1~53)",
}


def write_doc(frames):
    ds = {}
    for line in (ROOT / "docs" / "dataset.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 3 and cells[1] and cells[1] not in ds:
            ds[cells[1]] = cells[2]
    lines = ["# 학습용 피처 테이블 (features.parquet)", "",
             "> 자동 생성: `python scripts/build_features.py`. 원본 변수 정의는 [dataset.md](dataset.md)",
             "> 1행 = 시계열(품목 × 품종 × 등급 × 소매/중도매) × 가격 조사일. 결측은 채우지 않음, 스케일링 없음",
             "> 학습 2016~2022 / 테스트 2023~2024. 2015년은 가격 이력 계산용 준비 기간", ""]
    for d, df in frames.items():
        feats = [c for c in df.columns if c not in ID or c in CAT]
        feats = [c for c in feats if c not in ("y", "base")]
        lines += [f"## {NAMES[d]} — {len(df):,}행, 피처 {len(feats)}개", "",
                  "| 컬럼 | 역할 | 결측 % | 설명 |", "|---|---|---|---|"]
        na = df.isna().mean() * 100
        for c in df.columns:
            role = "타깃" if c == "y" else "보조" if c == "base" else "키" if c in ID and c not in CAT else "피처"
            lines.append(f"| {c} | {role} | {na[c]:.1f} | {DOC.get(c, ds.get(c, ''))} |")
        lines.append("")
    (ROOT / "docs" / "feature_set.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    frames = {}
    for d in NAMES:
        df = pd.read_parquet(P / d / "dataset.parquet")
        df = add_history_and_target(df)
        df["woy"] = df.date.dt.isocalendar().week.astype("int16")
        cols = ID + COMMON + DOMAIN[d] + HIST + ["woy", "base", "y"]
        missing = [c for c in cols if c not in df.columns]
        assert not missing, f"{d}: dataset.parquet에 없는 컬럼 {missing}"
        df = df.loc[df.y.notna(), cols].reset_index(drop=True)
        n_feat = len(COMMON) + len(DOMAIN[d]) + len(HIST) + 1 + len(CAT)
        df.to_parquet(P / d / "features.parquet", index=False)
        frames[d] = df
        print(f"{d}/features.parquet  {len(df):,}행  피처 {n_feat}개  y 결측 {df.y.isna().sum()}")
    write_doc(frames)
    print("저장: docs/feature_set.md")
