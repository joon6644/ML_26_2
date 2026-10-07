"""분야별 표준 분석용 데이터셋 → data/processed/{agri,livestock,fishery}/dataset.parquet

    python scripts/build_dataset.py      # build.py 다음에 실행

원칙 (데이터 수집 단계용):
- 1행 = 시계열(품목 × 품종 × 등급 × 소매/중도매) × 가격 조사일. 가격 = 그날 전국 시장 원/kg 중앙값 (친환경 제외)
- 변수는 원본 값 그대로 붙인다 (비율·이동평균 같은 가공 없음). 가공 피처는 모델링 단계에서 만든다.
- 일 단위 변수는 같은 날 값, 월·분기·연 통계는 '공개일'(기간 종료 + 공개 시차) 기준 그날까지 공개된 최신 값.
    공개 시차: 관세청 수입 월말+20일, 국제상품가격 월말+20일, 어업생산동향 월말+35일, 축평원 경락 월말+20일·재고 월말+45일,
              가축동향 분기말+60일, 농작물 생산량 다음 해 3월 1일
- 결측은 채우지 않는다 (NaN). 타깃 계산용 향후 14·28일 평균 가격을 함께 넣는다.
컬럼 설명: docs/dataset.md
"""
import numpy as np
import pandas as pd

from common import PROCESSED as P
from common import ROOT

REF = ROOT / "data" / "reference"
SHARE_YEARS = (2016, 2022)   # 주산지(시도별 생산 비중)·주 원산지를 정하는 기간


def item_map():
    return pd.read_csv(REF / "item_map.csv", dtype=str).fillna("")


def sido_alias():
    a = pd.read_csv(REF / "sido_alias.csv", dtype=str)
    return dict(zip(a.name, a.sido))


def month_end(x):
    """월(첫날 등) → 그 달 말일. Series·Index 어느 쪽이 와도 DatetimeIndex로 반환."""
    return (pd.DatetimeIndex(pd.to_datetime(x)) + pd.offsets.MonthEnd(0)).normalize()


def asof(df, feat, by, cols):
    """feat: [by..., avail, cols...] → 각 행 date 시점까지 공개된 최신 값."""
    if feat.empty:
        return df.assign(**{c: np.nan for c in cols})
    feat = feat.dropna(subset=["avail"]).sort_values("avail")
    left = df.reset_index().sort_values("date")
    m = pd.merge_asof(left, feat[by + ["avail"] + cols].rename(columns={"avail": "date"}), on="date", by=by or None, direction="backward")
    return m.set_index("index").sort_index()


def day_merge(df, feat, on):
    return df.merge(feat, on=on, how="left")


# ------------------------------------------------------------------ 기본: 가격 시계열과 타깃

def base_rows(domain):
    p = pd.read_parquet(P / domain / "price.parquet", columns=["date", "se_nm", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "price_kg"])
    p = p[p.se_nm.isin(["소매", "중도매"]) & (p.price_kg > 0)]
    keys = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
    daily = p.groupby(keys + ["date"], observed=True).price_kg.median().reset_index()
    out = []
    for _, g in daily.groupby(keys, observed=True):
        s = g.set_index("date").price_kg.asfreq("D")
        rev = s[::-1]
        f = g.copy()
        for h in (14, 28):                                                   # 향후 h일(t+1~t+h) 조사 가격 평균 = 타깃 재료
            fut = rev.rolling(h, min_periods=max(5, h // 3)).mean()[::-1].shift(-1)
            f[f"target_mean_{h}d"] = fut.reindex(g.date).values
        out.append(f)
    df = pd.concat(out, ignore_index=True)
    im = item_map()[["ctgry_cd", "item_cd", "food_group"]]
    return df.merge(im, on=["ctgry_cd", "item_cd"], how="left"), daily


def common_vars(df):
    cal = pd.read_parquet(P / "common" / "calendar.parquet")[["date", "is_holiday", "days_to_seollal", "days_to_chuseok"]]
    df = day_merge(df, cal, "date")
    df["year"], df["month"], df["dow"] = df.date.dt.year, df.date.dt.month, df.date.dt.dayofweek
    w = pd.read_parquet(P / "common" / "weather_national_daily.parquet")          # 전국 평균 기상 (전 항목)
    df = day_merge(df, w.rename(columns={c: f"wx_{c}" for c in w.columns if c != "date"}), "date")
    fx = pd.read_parquet(P / "common" / "fx_daily.parquet").assign(avail=lambda d: d.date)[["avail", "usd_krw"]]
    df = asof(df, fx, [], ["usd_krw"])                                       # 주말·휴일은 직전 영업일 환율

    # 월 거시지표 (국제상품가격 전 항목, 주유소 유가, 소비자물가 총지수·식료품, 수입물가지수 전 항목) - 공개일 기준
    m = pd.read_parquet(P / "common" / "macro_monthly.parquet")
    prefix = {"intl_commodity": "intl_", "fuel_price": "fuel_", "import_price": "impidx_"}
    sel = m[m.table.isin(prefix) | ((m.table == "cpi") & m.name.isin(["총지수", "식료품"]))].copy()
    sel["col"] = sel.table.map(prefix).fillna("cpi_") + sel.name.str.replace(r"[\s,\-/()]+", "_", regex=True).str.strip("_")
    wide = sel.pivot_table(index="month", columns="col", values="value")
    cols = list(wide.columns)
    wide = wide.reset_index().assign(avail=lambda d: month_end(d.month) + pd.Timedelta(days=20)).drop(columns="month")
    df = asof(df, wide, [], cols)

    # 품목 자신의 소비자물가지수 (CPI 품목명이 품목명과 같을 때)
    cpi = m[m.table == "cpi"]
    own = cpi[cpi.name.isin(df.item_nm.unique())].assign(avail=lambda d: month_end(d.month) + pd.Timedelta(days=20))
    df = asof(df, own.rename(columns={"name": "item_nm", "value": "cpi_item"})[["item_nm", "avail", "cpi_item"]], ["item_nm"], ["cpi_item"])

    # 품목 영양성분 (정적, item_map.nutrition_code)
    nu = pd.read_parquet(P / "common" / "nutrition_raw_material.parquet")
    keep = {"식품코드": "nutrition_code", "에너지(kcal)": "nutr_kcal", "단백질(g)": "nutr_protein_g", "지방(g)": "nutr_fat_g",
            "탄수화물(g)": "nutr_carb_g", "당류(g)": "nutr_sugar_g", "식이섬유(g)": "nutr_fiber_g", "나트륨(mg)": "nutr_sodium_mg", "폐기율(%)": "nutr_waste_pct"}
    nu = nu[list(keep)].rename(columns=keep).drop_duplicates("nutrition_code")
    for c in keep.values():
        if c != "nutrition_code":
            nu[c] = pd.to_numeric(nu[c], errors="coerce")
    link = item_map()[["ctgry_cd", "item_cd", "nutrition_code"]].merge(nu, on="nutrition_code", how="left").drop(columns="nutrition_code")
    return df.merge(link, on=["ctgry_cd", "item_cd"], how="left")


def mafra_wholesale(df, domain):
    """MAFRA 전국 도매시장 가격 → 품목별 그날 원/kg 중앙값 (단위가 'NNkg'인 행만 환산)."""
    path = P / domain / "wholesale_mafra.parquet"
    if not path.exists():
        return df.assign(mafra_whsl_price_kg=np.nan)
    w = pd.read_parquet(path, columns=["date", "ctgry_cd", "item_cd", "unit", "price"])
    kg = pd.to_numeric(w.unit.str.extract(r"^([\d.]+)\s*kg$", expand=False), errors="coerce")
    w = w.assign(p=w.price / kg).dropna(subset=["p"])
    w = w[w.p > 0].groupby(["date", "ctgry_cd", "item_cd"]).p.median().rename("mafra_whsl_price_kg").reset_index()
    return day_merge(df, w, ["date", "ctgry_cd", "item_cd"])


def import_kg(df):
    """품목별 HS 수입 중량 (item_map.hs_prefixes), 공개된 최신 월 값과 그 월."""
    imp = pd.concat([pd.read_parquet(P / d / "import_monthly.parquet") for d in ("agri", "livestock", "fishery")], ignore_index=True)
    rows = []
    for r in item_map().itertuples():
        pre = tuple(x for x in r.hs_prefixes.split(";") if x)
        if not pre:
            continue
        s = imp[imp.hs_code.str.startswith(pre)].groupby("month")[["imp_kg", "imp_usd", "exp_kg"]].sum()
        if s.empty:
            continue
        rows.append(pd.DataFrame({"ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd, "avail": month_end(s.index) + pd.Timedelta(days=20),
                                  "import_kg": s.imp_kg.values, "import_usd": s.imp_usd.values, "export_kg": s.exp_kg.values, "import_month": s.index}))
    return asof(df, pd.concat(rows), ["ctgry_cd", "item_cd"], ["import_kg", "import_usd", "export_kg", "import_month"])


def wholesale_price(df, daily):
    """같은 품목의 그날 중도매 가격 (품목 내 중도매 시계열들의 원/kg 중앙값)."""
    w = daily[daily.se_nm == "중도매"].groupby(["ctgry_cd", "item_cd", "date"]).price_kg.median().rename("whsl_price_kg").reset_index()
    return day_merge(df, w, ["ctgry_cd", "item_cd", "date"])


# ------------------------------------------------------------------ 농산

def agri_vars(df, daily):
    t = pd.read_parquet(P / "agri" / "trade_by_item_daily.parquet")[["date", "ctgry_cd", "item_cd", "qty_kg", "avg_price_kg", "amount", "n_trades"]]
    t = t.rename(columns={"qty_kg": "garak_qty_kg", "avg_price_kg": "garak_price_kg", "amount": "garak_amount_krw", "n_trades": "garak_n_trades"})
    df = day_merge(df, t, ["date", "ctgry_cd", "item_cd"])
    no_trade = (df.date >= "2018-01-03") & df.garak_qty_kg.isna() & df.item_cd.isin(t.item_cd.unique())
    df.loc[no_trade, ["garak_qty_kg", "garak_amount_krw", "garak_n_trades"]] = 0                       # 정산 기록이 없는 날 = 거래 0

    # 가락 반입 중 주 원산지(SHARE_YEARS 물량 1위 시도) 비중 (그날 기준)
    o_path = P / "agri" / "trade_origin_daily.parquet"
    if o_path.exists():
        o = pd.read_parquet(o_path)
        parts = []
        for r in item_map().itertuples():
            for k in [x for x in r.garak_item.split(";") if x]:
                g = o[o.garak_key == k]
                b = g[g.date.dt.year.between(*SHARE_YEARS) & (g.sido != "기타")]
                if b.empty:
                    continue
                top = b.groupby("sido").qty.sum().idxmax()
                tot = g.groupby("date").qty.sum()
                parts.append(pd.DataFrame({"date": tot.index, "ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd, "garak_top_origin": top,
                                           "garak_top_origin_share": (g[g.sido == top].groupby("date").qty.sum().reindex(tot.index).fillna(0) / tot).values}))
        if parts:
            f = pd.concat(parts).groupby(["date", "ctgry_cd", "item_cd"]).agg(garak_top_origin=("garak_top_origin", "first"),
                                                                               garak_top_origin_share=("garak_top_origin_share", "mean")).reset_index()
            df = day_merge(df, f, ["date", "ctgry_cd", "item_cd"])

    # 주산지 가중 일 기상 (SHARE_YEARS 시도별 생산량 비중 × 시도 평균)
    alias = sido_alias()
    prod = pd.read_parquet(P / "agri" / "production_annual.parquet")
    prod = prod[prod.item_nm.str.endswith(":생산량")].assign(sido=lambda d: d["시도별"].map(alias).replace({"전남+광주": "전남"}),
                                                               year=lambda d: d.period.astype(int))
    st = pd.read_parquet(P / "common" / "weather_station_daily.parquet")
    st = st.merge(pd.read_csv(REF / "asos_station_sido.csv")[["stnId", "sido"]].rename(columns={"stnId": "stn_id"}).astype({"stn_id": str}), on="stn_id")
    piv = {c: st.groupby(["sido", "date"])[c].mean().unstack("sido").asfreq("D") for c in ["temp_avg", "temp_max", "temp_min", "rain", "sunshine_hr", "humidity", "wind_avg"]}
    regional = prod[prod.sido.notna() & (prod.sido != "전국") & prod.year.between(*SHARE_YEARS)]
    area_all = pd.read_parquet(P / "agri" / "production_annual.parquet")
    area_all = area_all[area_all.item_nm.str.endswith(":면적") & area_all["시도별"].isin(["계", "전국"])].assign(year=lambda d: d.period.astype(int))
    parts, nat_rows = [], []
    for r in item_map().itertuples():
        if r.domain != "agri" or ":" not in r.kosis_production:
            continue
        tbl, name = r.kosis_production.split(":")
        p = regional[(regional.table == tbl) & (regional.item_nm == f"{name}:생산량")]
        if not p.empty:
            wts = p.groupby("sido").value.sum()
            wts = wts[(wts > 0) & wts.index.isin(piv["rain"].columns)]
            wts = wts / wts.sum()
            ws = {c: (v[wts.index] * wts).sum(axis=1, min_count=1) / (v[wts.index].notna() * wts).sum(axis=1).replace(0, np.nan) for c, v in piv.items()}
            parts.append(pd.DataFrame({"date": piv["rain"].index, "ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd,
                                       **{f"prodarea_{c}": ws[c].values for c in ws}}))
        nat = prod[(prod.table == tbl) & (prod.item_nm == f"{name}:생산량") & prod["시도별"].isin(["계", "전국"])].groupby("year").value.sum()
        area = area_all[(area_all.table == tbl) & (area_all.item_nm == f"{name}:면적")].groupby("year").value.sum()
        if len(nat):
            nat_rows.append(pd.DataFrame({"ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd, "avail": pd.to_datetime([f"{y + 1}-03-01" for y in nat.index]),
                                          "production_ton": nat.values, "cultivated_area_ha": area.reindex(nat.index).values, "production_year": nat.index}))
    if parts:
        df = day_merge(df, pd.concat(parts), ["date", "ctgry_cd", "item_cd"])
    if nat_rows:
        df = asof(df, pd.concat(nat_rows), ["ctgry_cd", "item_cd"], ["production_ton", "cultivated_area_ha", "production_year"])
    df = mafra_wholesale(import_kg(df), "agri")
    return wholesale_price(df, daily)


# ------------------------------------------------------------------ 축산

CENSUS = {"소": ("livestock_census", "한우:마리수"), "돼지": ("pig", "마리수"), "계란": ("chicken_census", "산란계:마리수"),
          "닭": ("chicken_census", "육용계:마리수"), "우유": ("livestock_census", "젖소:마리수")}
SPECIES = {"소": "소", "수입 소고기": "소", "돼지": "돼지", "수입 돼지고기": "돼지"}
DISEASE = {"계란": "고병원성조류인플루엔자", "닭": "고병원성조류인플루엔자", "돼지": "아프리카돼지열병", "수입 돼지고기": "아프리카돼지열병",
           "소": "구제역", "수입 소고기": "구제역", "우유": "구제역"}


def livestock_vars(df, daily):
    d = pd.read_parquet(P / "livestock" / "disease_events.parquet")
    ev = d.groupby(["disease_std", "date"]).agg(disease_cases=("disease", "size"), disease_heads=("head_count", "sum")).reset_index()
    df["disease_std"] = df.item_nm.map(DISEASE)
    df = df.merge(ev, on=["disease_std", "date"], how="left")
    df[["disease_cases", "disease_heads"]] = df[["disease_cases", "disease_heads"]].fillna(0)   # 발생 없음 = 0
    alld = d.groupby("date").agg(disease_all_cases=("disease", "size"), disease_all_heads=("head_count", "sum")).reset_index()
    df = day_merge(df, alld, "date")                                                            # 모든 법정 가축전염병 (전 축종)
    df[["disease_all_cases", "disease_all_heads"]] = df[["disease_all_cases", "disease_all_heads"]].fillna(0)

    ce = pd.read_parquet(P / "livestock" / "census_quarterly.parquet")
    ce = ce[(ce["시도별"] == "전국") & (ce["사육규모별"].fillna("합계") == "합계")]
    rows = []
    for item, (tbl, nm) in CENSUS.items():
        s = ce[ce.table.isin(["pig_census_old", "pig_census"]) & (ce.item_nm == nm)].sort_values("table").drop_duplicates("period", keep="last") \
            if tbl == "pig" else ce[(ce.table == tbl) & (ce.item_nm == nm)]
        s = s.groupby("period").value.sum()
        farm_src = ce[ce.table.isin(["pig_census_old", "pig_census"])] if tbl == "pig" else ce[ce.table == tbl]
        farms = farm_src[farm_src.item_nm.isin([nm.replace("마리수", "농장수"), nm.replace("마리수", "가구수")])].groupby("period").value.sum()
        q_end = pd.PeriodIndex([f"{p[:4]}Q{int(p[4:])}" for p in s.index], freq="Q").end_time.normalize()
        rows.append(pd.DataFrame({"item_nm": item, "avail": q_end + pd.Timedelta(days=60), "census_heads": s.values,
                                  "census_farms": farms.reindex(s.index).values, "census_period": s.index}))
    df = asof(df, pd.concat(rows), ["item_nm"], ["census_heads", "census_farms", "census_period"])

    a = pd.read_parquet(P / "livestock" / "auction_monthly.parquet")
    a = a[(a.market_cd == "CTot") & (a.grade == "평균")]
    s = pd.read_parquet(P / "livestock" / "stock_monthly.parquet")
    arows, srows = [], []
    for item, sp in SPECIES.items():
        g = a[a.species == sp]
        arows.append(pd.DataFrame({"item_nm": item, "avail": month_end(g.month) + pd.Timedelta(days=20),
                                   "auction_heads": g.head_count.values, "auction_price_kg": g.price.values}))
        g = s[s.species == sp]
        srows.append(pd.DataFrame({"item_nm": item, "avail": month_end(g.month) + pd.Timedelta(days=45), "stock_ton": g.totStock.values}))
    df = asof(df, pd.concat(arows), ["item_nm"], ["auction_heads", "auction_price_kg"])
    df = asof(df, pd.concat(srows), ["item_nm"], ["stock_ton"])

    pr = pd.read_parquet(P / "livestock" / "pig_rep_price_daily.parquet")
    pr = pr.pivot_table(index="date", columns="skin", values="price").rename(columns={"탕박": "pig_auction_price_kg", "박피": "pig_auction_skinned_kg",
                                                                                 "대표가격": "pig_auction_rep_kg"})
    pcols = list(pr.columns)
    df = asof(df, pr.reset_index().rename(columns={"date": "avail"}), [], pcols)   # 돼지 도체 경락가, 직전 경매일 값
    df = import_kg(df)
    return wholesale_price(df, daily).drop(columns="disease_std")


# ------------------------------------------------------------------ 수산

def fishery_vars(df, daily):
    sea = pd.read_parquet(P / "fishery" / "sea_temp_daily.parquet").pivot(index="date", columns="sea", values="water_temp")
    sea = sea.rename(columns={"동해": "sea_temp_east", "남해": "sea_temp_south", "서해": "sea_temp_west"}).reset_index()
    df = day_merge(df, sea, "date")
    fp = pd.read_parquet(P / "fishery" / "production_monthly.parquet")
    fp = fp[(fp["어업별"] == "계") & fp.item_id.isin(["T01", "T05"])]
    rows = []
    for r in item_map().itertuples():
        if not r.kosis_production.startswith("fishery:"):
            continue
        g = fp[fp["품종별"] == r.kosis_production.split(":", 1)[1]].pivot_table(index="period", columns="item_id", values="value", aggfunc="sum")
        if g.empty:
            continue
        mi = pd.to_datetime(g.index, format="%Y%m")
        rows.append(pd.DataFrame({"ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd, "avail": month_end(mi) + pd.Timedelta(days=35),
                                  "catch_ton": g.get("T01", pd.Series(np.nan, index=g.index)).values,
                                  "catch_value_kkrw": g.get("T05", pd.Series(np.nan, index=g.index)).values, "catch_month": mi}))
    df = asof(df, pd.concat(rows), ["ctgry_cd", "item_cd"], ["catch_ton", "catch_value_kkrw", "catch_month"])
    rt = pd.read_parquet(P / "fishery" / "redtide_events.parquet").dropna(subset=["date"])
    rt = rt.groupby("date").agg(redtide_reports=("news_id", "nunique"), redtide_areas=("area", "nunique"),
                                redtide_density_max=("density_max", "max")).reset_index()
    df = day_merge(df, rt, "date")
    df[["redtide_reports", "redtide_areas"]] = df[["redtide_reports", "redtide_areas"]].fillna(0)   # 보고 없음 = 0
    jf = pd.read_parquet(P / "fishery" / "jellyfish_reports.parquet").groupby("date").size().rename("jellyfish_reports").reset_index()
    df = day_merge(df, jf, "date")
    df["jellyfish_reports"] = df.jellyfish_reports.fillna(0)                 # 해파리 모니터링 주간보고 발행 (5~12월)
    b = pd.read_parquet(P / "fishery" / "buoy_temp_daily.parquet")
    b = b[b.core & (b.n_hours >= 12)].groupby("date").agg(buoy_wind_avg=("wind", "mean"), buoy_temp_max=("water_temp_max", "max")).reset_index()
    df = day_merge(df, b, "date")
    ct = pd.read_parquet(P / "fishery" / "coast_temp_daily.parquet").dropna(subset=["water_temp"])
    ct = ct.groupby("date").agg(coast_temp_avg=("water_temp", "mean"), coast_temp_n_stations=("stn_cd", "nunique")).reset_index()
    df = day_merge(df, ct, "date")                                           # 연안정지관측 (관측소가 17곳 → 1곳으로 줄어듦, n 확인)
    df = closed_season(df)
    df = farm_env(df)
    df = mafra_wholesale(import_kg(df), "fishery")
    return wholesale_price(df, daily)


FARM_VARS = ["temp_s", "temp_b", "sal_s", "do_s", "do_b", "cod_s", "din_s", "dip_s", "chl_s", "ss_s"]
FARM_KIND = {"굴": "굴", "홍합": "진주담치", "바지락": "바지락", "가리비": "가리비", "김": "김", "마른미역": "미역|다시마|해조",
             "건다시마": "미역|다시마|해조", "전복": "전복"}                      # 품목 → 어장환경 조사의 양식 품종(kind) 패턴


def farm_env(df):
    """수산과학원 어장환경 조사(양식장 수질): 조사 월별 평균을 월말+30일 공개로 보고 붙인다. 전체 평균(farm_) + 품목 어장 평균(farmitem_)."""
    path = P / "fishery" / "farm_env_survey.parquet"
    if not path.exists():
        return df
    f = pd.read_parquet(path).dropna(subset=["date"])
    f["month"] = f.date.dt.to_period("M").dt.to_timestamp()
    allm = f.groupby("month")[FARM_VARS].mean().add_prefix("farm_").reset_index()
    allm["avail"] = month_end(allm.month) + pd.Timedelta(days=30)
    df = asof(df, allm.rename(columns={"month": "farm_survey_month"}), [], [f"farm_{v}" for v in FARM_VARS] + ["farm_survey_month"])
    rows = []
    for item, pat in FARM_KIND.items():
        g = f[f.kind.fillna("").str.contains(pat)].groupby("month")[FARM_VARS].mean().add_prefix("farmitem_").reset_index()
        g["avail"] = month_end(g.month) + pd.Timedelta(days=30)
        rows.append(g.drop(columns="month").assign(item_nm=item))
    return asof(df, pd.concat(rows), ["item_nm"], [f"farmitem_{v}" for v in FARM_VARS])


def closed_season(df):
    """금어기 (data/reference/closed_season.csv, 2026.1.1 기준 규정). closed_season = 그날 금어기(확정), closed_window = 규정상 가능 기간,
    days_to_closed_start = 다음 금어기 시작까지 일수. 고등어 2023년 이전은 고시 날짜를 몰라 closed_season을 비워 둔다."""
    cs = pd.read_csv(REF / "closed_season.csv", dtype=str).fillna("")
    md = df.date.dt.strftime("%m-%d")
    df["closed_season"], df["closed_window"], df["days_to_closed_start"] = 0.0, 0.0, np.nan
    for item, g in cs.groupby("item_nm"):
        m = df.item_nm == item
        if not m.any():
            continue
        generic = g[g.year == ""].iloc[0]
        win = (md[m] >= generic.start_md) & (md[m] <= generic.end_md)
        df.loc[m, "closed_window"] = win.astype(float).values
        yearly = g[g.year != ""]
        if len(yearly):                                                       # 연도별 고시 품목 (고등어)
            yrs = df.loc[m, "date"].dt.year.astype(str)
            conf = pd.Series(np.nan, index=df.index[m])
            for r in yearly.itertuples():
                sel = yrs == r.year
                conf[sel] = ((md[m][sel] >= r.start_md) & (md[m][sel] <= r.end_md)).astype(float)
            df.loc[m, "closed_season"] = conf.values
        else:
            df.loc[m, "closed_season"] = win.astype(float).values
        d = df.loc[m, "date"]
        start_md = {r.year: r.start_md for r in yearly.itertuples()}           # 고시일을 아는 연도는 실제 시작일 사용

        def start_of(year):
            return pd.to_datetime(year.astype(str) + "-" + year.astype(str).map(start_md).fillna(generic.start_md))
        start = start_of(d.dt.year)
        start = start.where(start >= d, start_of(d.dt.year + 1))
        df.loc[m, "days_to_closed_start"] = (start - d).dt.days.values
    return df


# ------------------------------------------------------------------ 컬럼 사전 (docs/dataset.md 자동 생성)

DESC = {
    "domain": ("키", "분야 (agri / livestock / fishery)"), "date": ("키", "가격 조사일"),
    "ctgry_cd": ("키", "부류 코드 (100 식량 ~ 600 수산)"), "item_cd": ("키", "품목 코드"), "item_nm": ("키", "품목명"),
    "vrty_nm": ("키", "품종명"), "grd_nm": ("키", "등급명"), "se_nm": ("키", "소매 / 중도매"), "food_group": ("품목 속성", "식재료 계열 (item_map)"),
    "price_kg": ("가격", "그날 전국 시장 가격의 중앙값 (원/kg, 단위 환산 후)"),
    "target_mean_14d": ("타깃 재료", "향후 14일(t+1~t+14) 조사 가격 평균 (원/kg). 미래 값이므로 피처로 쓰면 안 됨"),
    "target_mean_28d": ("타깃 재료", "향후 28일(4주, t+1~t+28) 조사 가격 평균 (원/kg). 미래 값이므로 피처로 쓰면 안 됨"),
    "is_holiday": ("달력", "공휴일 여부"), "days_to_seollal": ("달력", "다음 설날까지 남은 일수"), "days_to_chuseok": ("달력", "다음 추석까지 남은 일수"),
    "year": ("달력", "연"), "month": ("달력", "월"), "dow": ("달력", "요일 (0=월)"),
    "usd_krw": ("경제", "원/달러 환율 (주말·휴일은 직전 영업일)"),
    "cpi_item": ("경제(월)", "품목 자신의 소비자물가지수 (2020=100, CPI 품목명이 같을 때만)"),
    "import_kg": ("수입(월)", "품목 HS코드 수입 중량 kg (공개된 최신 월)"), "import_usd": ("수입(월)", "품목 HS코드 수입 금액 USD"),
    "export_kg": ("수입(월)", "품목 HS코드 수출 중량 kg"), "import_month": ("수입(월)", "위 수입 값의 기준 월"),
    "whsl_price_kg": ("가격 신호", "같은 품목의 그날 aT 중도매 가격 (원/kg, 품목 내 중앙값)"),
    "mafra_whsl_price_kg": ("가격 신호", "같은 품목의 그날 MAFRA 전국 도매시장 가격 (원/kg, 단위가 kg인 거래만)"),
    "garak_qty_kg": ("공급: 가락", "가락시장 그날 정산 물량 (kg, 정산 없는 날 0, 2018-01~)"), "garak_price_kg": ("공급: 가락", "가락시장 그날 평균 정산가 (원/kg)"),
    "garak_amount_krw": ("공급: 가락", "가락시장 그날 정산 금액 (원)"), "garak_n_trades": ("공급: 가락", "가락시장 그날 정산 건수"),
    "garak_top_origin": ("공급: 가락", "품목의 주 원산지 시도 (2016~2022 물량 1위)"), "garak_top_origin_share": ("공급: 가락", "그날 가락 반입 중 주 원산지 비중 (0~1)"),
    "production_ton": ("공급: 생산(연)", "전국 생산량 톤 (Y년 값은 Y+1년 3월부터 공개로 간주)"), "cultivated_area_ha": ("공급: 생산(연)", "전국 재배면적 ha"),
    "production_year": ("공급: 생산(연)", "위 생산 값의 기준 연도"),
    "disease_cases": ("공급: 질병", "그날 품목 관련 주요 가축질병 발생 건수 (계란·닭=고병원성 AI, 돼지=ASF, 소·우유=구제역)"),
    "disease_heads": ("공급: 질병", "그날 위 질병 발생 두수"), "disease_all_cases": ("공급: 질병", "그날 모든 법정 가축전염병 발생 건수"),
    "disease_all_heads": ("공급: 질병", "그날 모든 법정 가축전염병 발생 두수"),
    "census_heads": ("공급: 사육(분기)", "사육 마리수 (소=한우, 돼지, 계란=산란계, 닭=육용계, 우유=젖소, 분기말+60일 공개)"),
    "census_farms": ("공급: 사육(분기)", "사육 농장(가구) 수"), "census_period": ("공급: 사육(분기)", "위 사육 값의 기준 분기 (YYYYQQ)"),
    "auction_heads": ("공급: 경락(월)", "소·돼지 전국 도체 경락 두수 (월, 평균 등급)"), "auction_price_kg": ("공급: 경락(월)", "소·돼지 전국 도체 경락가 원/kg (월)"),
    "stock_ton": ("공급: 재고(월)", "축산물 재고 톤 (2019-05~)"),
    "pig_auction_price_kg": ("가격 신호", "돼지 도체 경락가 탕박 원/kg (직전 경매일)"), "pig_auction_skinned_kg": ("가격 신호", "돼지 도체 경락가 박피 원/kg"),
    "pig_auction_rep_kg": ("가격 신호", "돼지 도체 대표가격 원/kg"),
    "sea_temp_east": ("수산 환경", "동해 수온 ℃ (기상청 부이, 2016년부터 운영된 부이 평균)"), "sea_temp_south": ("수산 환경", "남해 수온 ℃"),
    "sea_temp_west": ("수산 환경", "서해 수온 ℃"), "buoy_wind_avg": ("수산 환경", "부이 평균 풍속 m/s"), "buoy_temp_max": ("수산 환경", "부이 일 최고 수온 ℃"),
    "coast_temp_avg": ("수산 환경", "연안정지관측 평균 수온 ℃ (관측소가 17곳 → 1곳으로 줄어듦)"), "coast_temp_n_stations": ("수산 환경", "그날 연안정지관측 관측소 수"),
    "catch_ton": ("공급: 어획(월)", "품종 어업 생산량 톤 (월말+35일 공개)"), "catch_value_kkrw": ("공급: 어획(월)", "품종 어업 생산금액 천원"),
    "catch_month": ("공급: 어획(월)", "위 어획 값의 기준 월"),
    "redtide_reports": ("수산 환경", "그날 적조 발생 보고 수"), "redtide_areas": ("수산 환경", "그날 적조 발생 해역 수"),
    "redtide_density_max": ("수산 환경", "그날 적조 최대 밀도 (개체/mL)"), "jellyfish_reports": ("수산 환경", "그날 해파리 모니터링 주간보고 발행 수 (5~12월)"),
    "closed_season": ("수산 규제", "그날 금어기 여부 (1/0). 고등어 2023년 이전은 고시일 미상이라 비어 있음 (data/reference/closed_season.csv)"),
    "closed_window": ("수산 규제", "규정상 금어기가 들 수 있는 기간 여부 (고등어 = 4/1~6/30, 나머지 = 금어기와 같음)"),
    "days_to_closed_start": ("수산 규제", "다음 금어기 시작까지 남은 일수 (금어기 없는 품목은 비어 있음)"),
    "farm_survey_month": ("수산 환경(어장)", "어장환경 조사 값의 기준 월 (조사 월말+30일 공개로 간주)"),
}
PREFIX = {"wx_": ("기상(전국)", "ASOS 97개 관측소 평균 일기상"), "prodarea_": ("공급: 주산지 기상", "품목 주산지(시도별 생산 비중 가중) 일기상"),
          "intl_": ("경제(월)", "국제상품가격 (ECOS, 원유·곡물·금속 등, 월말+20일 공개)"), "fuel_": ("경제(월)", "주유소 평균 판매가격 원/ℓ (KOSIS)"),
          "cpi_": ("경제(월)", "소비자물가지수 (2020=100)"), "impidx_": ("경제(월)", "수입물가지수 (ECOS, 품목군별)"),
          "nutr_": ("품목 속성", "품목 대표 식품의 영양성분 (100g당, 정적)"),
          "farmitem_": ("수산 환경(어장)", "해당 품목 양식 어장의 수질 조사 평균 (temp 수온, sal 염분, do 용존산소, cod, din 용존무기질소, dip 용존무기인, chl 엽록소, ss 부유물질 / _s 표층, _b 저층)"),
          "farm_": ("수산 환경(어장)", "전체 양식 어장 수질 조사 평균 (변수 약어는 farmitem_과 같음)")}


def write_doc(frames):
    lines = ["# 표준 분석용 데이터셋 (data/processed/{agri,livestock,fishery}/dataset.parquet)", "",
             "> 자동 생성: `python scripts/build_dataset.py` · 원칙은 스크립트 상단 docstring 참고", "",
             "- **1행 = 시계열(품목 × 품종 × 등급 × 소매/중도매) × 가격 조사일**. 변수는 원본 값 그대로 (가공 피처 없음)",
             "- 일 변수는 같은 날 값, 월·분기·연 통계는 **그날까지 공개된 최신 값** (공개 시차 반영) → 어떤 피처를 만들어도 미래 누수 없음",
             "- 결측은 채우지 않음. `target_mean_*`는 타깃 계산용 미래 값이므로 **피처로 쓰면 안 됨**", ""]
    for d, df in frames.items():
        na = df.isna().mean() * 100
        rows = []
        for c in df.columns:
            g, txt = DESC.get(c, next((v for k, v in PREFIX.items() if c.startswith(k)), ("기타", "")))
            rows.append((g, c, txt, f"{na[c]:.1f}"))
        t = pd.DataFrame(rows, columns=["구분", "컬럼", "설명", "결측%"])
        lines += [f"## {d} ({len(df):,}행 × {df.shape[1]}열, {df.date.min():%Y-%m-%d} ~ {df.date.max():%Y-%m-%d})", "",
                  t.to_markdown(index=False, disable_numparse=True), ""]
    (ROOT / "docs" / "dataset.md").write_text(chr(10).join(lines), encoding="utf-8")


DOMAIN_VARS = {"agri": agri_vars, "livestock": livestock_vars, "fishery": fishery_vars}
# SHAP(eda/09_shap.py)에서 4회(분할 A·B × LightGBM·XGBoost) 모두 비중 0.1% 미만이고, 해당 분야와 무관해 보이는 변수만 제거 (보수적)
# 날씨·질병·적조·해파리·금어기처럼 관련 있어 보이는 변수는 기여가 작아도 남긴다
SHAP_DROP = {
    "agri": ["is_holiday",                                                    # 가격 조사일은 공휴일이 아니라 항상 0
             "impidx_육가공품및낙농품", "impidx_떡_과자및면류", "impidx_수산물가공품"],
    "livestock": ["nutr_fiber_g", "nutr_sugar_g",                            # 육류·계란에는 거의 0
                  "impidx_떡_빵및과자류", "impidx_조미료및유지"],
    "fishery": ["is_holiday", "nutr_fiber_g",                                # 항상 0
                "impidx_떡_빵및과자류", "impidx_떡_과자및면류", "impidx_육가공품및낙농품", "impidx_조미료및유지"],
}

if __name__ == "__main__":
    frames = {}
    for domain, fn in DOMAIN_VARS.items():
        df, daily = base_rows(domain)
        df = fn(common_vars(df), daily).sort_values(["ctgry_cd", "item_cd", "vrty_nm", "grd_nm", "se_nm", "date"]).reset_index(drop=True)
        df.insert(0, "domain", domain)
        df = df.loc[:, df.notna().any()]                                     # 전부 결측인 열 제거 (예: 축산은 중도매가가 없음)
        df = df.drop(columns=SHAP_DROP[domain])
        df.to_parquet(P / domain / "dataset.parquet", index=False)
        frames[domain] = df
        print(f"{domain}/dataset.parquet  {len(df):,}행  {df.shape[1]}열  {df.date.min():%Y-%m-%d}~{df.date.max():%Y-%m-%d}")
    write_doc(frames)
    print("저장: docs/dataset.md")
