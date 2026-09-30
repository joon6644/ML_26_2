"""data/raw/ 조각들을 합쳐 data/processed/{common,agri,livestock,fishery}/*.parquet 로 만든다.

    python scripts/build.py

수집이 덜 된 소스는 있는 만큼만 합친다 (다시 실행하면 덮어씀). 컬럼 설명은 docs/processed_data.md.
"""
import glob
import re

import numpy as np
import pandas as pd

from common import PROCESSED, RAW, START

DOMAIN = {"100": "agri", "200": "agri", "300": "agri", "400": "agri", "500": "livestock", "600": "fishery"}


def read_all(pattern):
    fs = sorted(glob.glob(str(RAW / pattern), recursive=True))
    dfs = [pd.read_parquet(f) for f in fs]
    dfs = [d for d in dfs if len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", "").str.strip().replace({"": np.nan, "-": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")


def write(df, domain, name):
    if df is None or df.empty:
        print(f"  (건너뜀) {domain}/{name}: 데이터 없음")
        return
    path = PROCESSED / domain / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  {domain}/{name}.parquet  {len(df):>10,}행  {df.shape[1]}열")


def split_domain(df, col, name):
    for dom in ("agri", "livestock", "fishery"):
        write(df[df[col].map(DOMAIN) == dom].reset_index(drop=True), dom, name)


# ---------------------------------------------------------------- 가격

def price():
    df = read_all("at_price/*.parquet")
    if df.empty:
        return
    out = pd.DataFrame({
        "date": pd.to_datetime(df.exmn_ymd, format="%Y%m%d"),
        "se_cd": df.se_cd, "se_nm": df.se_nm,
        "ctgry_cd": df.ctgry_cd, "ctgry_nm": df.ctgry_nm,
        "item_cd": df.item_cd, "item_nm": df.item_nm,
        "vrty_cd": df.vrty_cd, "vrty_nm": df.vrty_nm,
        "grd_cd": df.grd_cd, "grd_nm": df.grd_nm,
        "sgg_cd": df.sgg_cd, "sgg_nm": df.sgg_nm,
        "mrkt_cd": df.mrkt_cd, "mrkt_nm": df.mrkt_nm,
        "unit": df.unit, "unit_sz": num(df.unit_sz),
        "price": num(df.exmn_dd_prc), "price_kg_api": num(df.exmn_dd_cnvs_prc),
    }).drop_duplicates()
    out = out[out.date >= START].sort_values(["ctgry_cd", "item_cd", "date"])
    out["unit_g"], out["unit_g_source"] = unit_grams(out)
    out["price_kg"] = out.price / (out.unit_sz * out.unit_g / 1000)
    split_domain(out, "ctgry_cd", "price")


def unit_grams(df):
    """단위 1개의 무게(g). 무게 단위는 그대로, 개수 단위(포기·개·마리 …)는 data/reference/unit_weight.csv로 환산.
    API의 exmn_dd_cnvs_prc는 개수 단위에서 환산되지 않은 값이 들어 있어 쓰지 않는다."""
    table = pd.read_csv(RAW.parent / "reference" / "unit_weight.csv", dtype={"vrty_contains": str}).fillna({"vrty_contains": ""})
    keys = df[["item_nm", "vrty_nm", "unit"]].drop_duplicates()
    res = {}
    for k in keys.itertuples(index=False):
        u = str(k.unit)
        if u.startswith("kg") or u in ("L", "ℓ"):  # "kg(그물망 3포기)"처럼 괄호 설명이 붙은 kg도 무게 단위
            res[k] = (1000.0, "weight")
        elif u in ("g", "ml"):
            res[k] = (1.0, "weight")
        else:
            cand = table[(table.item_nm == k.item_nm) & (table.unit == u)]
            hit = cand[cand.vrty_contains.map(lambda s: bool(s) and s in str(k.vrty_nm))]
            row = hit.iloc[0] if len(hit) else (cand[cand.vrty_contains == ""].iloc[0] if (cand.vrty_contains == "").any() else None)
            res[k] = (float(row.g_per_unit), f"table:{row.source}") if row is not None else (np.nan, "missing")
    idx = list(zip(df.item_nm, df.vrty_nm, df.unit))
    return [res[i][0] for i in idx], [res[i][1] for i in idx]


def mafra_wholesale():
    df = read_all("mafra_wholesale/**/*.parquet")
    if df.empty:
        return
    out = pd.DataFrame({
        "date": pd.to_datetime(df.EXAMIN_DE, format="%Y%m%d"),
        "ctgry_cd": df.FRMPRD_CATGORY_CD, "ctgry_nm": df.FRMPRD_CATGORY_NM,
        "item_cd": df.PRDLST_CD, "item_nm": df.PRDLST_NM, "vrty_cd": df.SPCIES_CD, "vrty_nm": df.SPCIES_NM,
        "grd_cd": df.GRAD_CD, "grd_nm": df.GRAD_NM, "unit": df.EXAMIN_UNIT,
        "area_cd": df.AREA_CD, "area_nm": df.AREA_NM, "mrkt_cd": df.MRKT_CD, "mrkt_nm": df.MRKT_NM,
        "price": num(df.AMT),
    }).drop_duplicates()
    split_domain(out, "ctgry_cd", "wholesale_mafra")


def at_trade():
    """가락시장 정산 거래를 일 × 품목(대/중/소분류) × 단위로 합산."""
    keys = ["trd_clcln_ymd", "gds_lclsf_cd", "gds_lclsf_nm", "gds_mclsf_cd", "gds_mclsf_nm", "gds_sclsf_cd", "gds_sclsf_nm", "unit_nm"]
    parts = []
    for f in sorted(glob.glob(str(RAW / "at_trade" / "*" / "*.parquet"))):  # 원본 1,500만 행이라 하루씩 집계해 메모리를 아낀다
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d = d[keys].assign(qty=num(d.unit_tot_qty).values, amt=num(d.totprc).values)
        parts.append(d.groupby(keys, dropna=False).agg(qty=("qty", "sum"), amount=("amt", "sum"), n_trades=("amt", "size")).reset_index())
    if not parts:
        return
    g = pd.concat(parts, ignore_index=True)
    g["avg_price"] = g.amount / g.qty
    g.insert(0, "date", pd.to_datetime(g.pop("trd_clcln_ymd")))
    write(g, "agri", "trade_garak_daily")

    # aT 품목 기준 합계 (data/reference/item_map.csv 의 garak_item). 물량은 kg 단위 거래만 합산
    m = pd.read_csv(RAW.parent / "reference" / "item_map.csv", dtype=str).fillna("")
    link = [(r.ctgry_cd, r.item_cd, r.item_nm, k) for r in m.itertuples() for k in r.garak_item.split(";") if k]
    link = pd.DataFrame(link, columns=["ctgry_cd", "item_cd", "item_nm", "garak_key"])
    g["garak_key"] = g.gds_lclsf_nm + ">" + g.gds_mclsf_nm
    j = g.merge(link, on="garak_key")
    kg = j[j.unit_nm == "kg"]
    out = (j.groupby(["date", "ctgry_cd", "item_cd", "item_nm"]).agg(amount=("amount", "sum"), n_trades=("n_trades", "sum"))
           .join(kg.groupby(["date", "ctgry_cd", "item_cd", "item_nm"]).agg(qty_kg=("qty", "sum"), amount_kg=("amount", "sum")))
           .reset_index())
    out["avg_price_kg"] = out.amount_kg / out.qty_kg
    write(out.drop(columns="amount_kg"), "agri", "trade_by_item_daily")


# ---------------------------------------------------------------- 공통

def calendar():
    h = read_all("calendar/holidays.parquet")
    if h.empty:
        return
    h["date"] = pd.to_datetime(h.date)
    days = pd.DataFrame({"date": pd.date_range(START, h.date.max())})
    days = days.merge(h.groupby("date").name.agg(", ".join).reset_index(), how="left")
    days["is_holiday"] = days.name.notna()
    days["is_weekend"] = days.date.dt.weekday >= 5
    for key, col in [("Seollal|설날", "seollal"), ("Chuseok|추석", "chuseok")]:
        main = h[h.name.str.contains(key) & ~h.name.str.contains("전날|다음날|eve|day after|대체|alternative", case=False)]
        nxt = pd.merge_asof(days[["date"]], main[["date"]].sort_values("date").assign(next_day=lambda x: x.date),
                            on="date", direction="forward")
        days[f"days_to_{col}"] = (nxt.next_day - days.date).dt.days  # 다음 명절 당일까지 남은 일수
    write(days.rename(columns={"name": "holiday_name"}), "common", "calendar")


def weather():
    df = read_all("asos/*.parquet")
    if df.empty:
        return
    cols = {"avgTa": "temp_avg", "minTa": "temp_min", "maxTa": "temp_max", "sumRn": "rain", "hr1MaxRn": "rain_1h_max",
            "sumSsHr": "sunshine_hr", "sumGsr": "solar_rad", "avgRhm": "humidity", "maxInsWs": "wind_gust", "avgWs": "wind_avg",
            "ddMes": "snow_depth", "avgTs": "ground_temp"}
    out = pd.DataFrame({"date": pd.to_datetime(df.tm), "stn_id": df.stnId, "stn_nm": df.stnNm})
    for k, v in cols.items():
        out[v] = num(df[k]) if k in df else np.nan
    out["rain"] = out.rain.fillna(0)  # ASOS는 무강수일을 빈값으로 준다
    write(out, "common", "weather_station_daily")
    nat = out.groupby("date")[list(cols.values())].mean().reset_index()
    write(nat, "common", "weather_national_daily")


def ecos():
    frames = []
    for name in ("fx_usd", "cpi", "import_price", "intl_commodity"):
        df = read_all(f"ecos/{name}.parquet")
        if df.empty:
            continue
        frames.append(pd.DataFrame({"source": "ECOS", "table": name, "code": df.ITEM_CODE1, "name": df.ITEM_NAME1,
                                    "unit": df.UNIT_NAME, "period": df.TIME, "value": num(df.DATA_VALUE)}))
    if not frames:
        return
    df = pd.concat(frames)
    fx = df[df.table == "fx_usd"]
    write(pd.DataFrame({"date": pd.to_datetime(fx.period, format="%Y%m%d"), "usd_krw": fx.value.values}), "common", "fx_daily")
    fuel = read_all("kosis/fuel_price/*.parquet")
    if not fuel.empty:
        df = pd.concat([df, pd.DataFrame({"source": "KOSIS", "table": "fuel_price", "code": fuel.C1, "name": fuel.C1_NM,
                                          "unit": fuel.UNIT_NM, "period": fuel.PRD_DE, "value": num(fuel.DT)})])
    m = df[df.table != "fx_usd"].copy()
    m["month"] = pd.to_datetime(m.period, format="%Y%m")
    write(m.drop(columns="period").reset_index(drop=True), "common", "macro_monthly")


# ---------------------------------------------------------------- 도메인 피처

def customs():
    df = read_all("customs/*.parquet")
    if df.empty:
        return
    df = df[df.year.str.match(r"^\d{4}\.\d{2}$", na=False)]  # '총계' 행 제외
    out = pd.DataFrame({"month": pd.to_datetime(df.year, format="%Y.%m"), "hs_code": df.hsCode, "hs_name": df.statKor,
                        "imp_kg": num(df.impWgt), "imp_usd": num(df.impDlr), "exp_kg": num(df.expWgt), "exp_usd": num(df.expDlr)})
    ch, h4 = out.hs_code.str[:2], out.hs_code.str[:4]
    live = ch.isin(["02", "04"]) | h4.isin(["1601", "1602"])
    fish = (ch == "03") | h4.isin(["1603", "1604", "1605", "2501"])  # 천일염은 수산 부류(600)
    write(out[live].reset_index(drop=True), "livestock", "import_monthly")
    write(out[fish].reset_index(drop=True), "fishery", "import_monthly")
    write(out[~live & ~fish].reset_index(drop=True), "agri", "import_monthly")


def kosis_long(name):
    df = read_all(f"kosis/{name}/*.parquet")
    if df.empty:
        return df
    dims = sorted({c[:-3] for c in df.columns if re.fullmatch(r"C\d_NM", c)})
    out = pd.DataFrame({"period": df.PRD_DE, "item_id": df.ITM_ID, "item_nm": df.ITM_NM, "unit": df.UNIT_NM})
    for d in dims:
        out[f"{df[f'{d}_OBJ_NM'].iloc[0]}"] = df[f"{d}_NM"].values
    out["value"] = num(df.DT).values
    return out


def production():
    agri = pd.concat([kosis_long(n).assign(table=n) for n in ("veg_leaf", "veg_root", "veg_seasoning", "food_crops", "fruit")], ignore_index=True)
    write(agri, "agri", "production_annual")
    census = [kosis_long(n).assign(table=n) for n in ("livestock_census", "pig_census_old", "pig_census", "chicken_census")]
    write(pd.concat([c for c in census if not c.empty], ignore_index=True), "livestock", "census_quarterly")
    write(kosis_long("fishery_production"), "fishery", "production_monthly")


def livestock():
    d = read_all("mafra_disease/all.parquet")
    if not d.empty:
        write(pd.DataFrame({
            "date": pd.to_datetime(d.OCCRRNC_DE, format="%Y%m%d", errors="coerce"), "end_date": pd.to_datetime(d.CESSATION_DE, format="%Y%m%d", errors="coerce"),
            "disease": d.LKNTS_NM,
            # 표기 통일: 띄어쓰기 제거, 가금티프스→가금티푸스, 세부형(-생식기형 등) 제거
            "disease_std": d.LKNTS_NM.str.replace(" ", "").str.replace("티프스", "티푸스").str.split("-").str[0],
            "species": d.LVSTCKSPC_NM, "head_count": num(d.OCCRRNC_LVSTCKCNT),
            "region": d.FARM_LOCPLC.str.split().str[0], "address": d.FARM_LOCPLC, "legaldong_cd": d.FARM_LOCPLC_LEGALDONG_CODE,
        }).sort_values("date").reset_index(drop=True), "livestock", "disease_events")
    frames = []
    for kind in ("cattle_auction", "pig_auction"):
        a = read_all(f"ekape/{kind}/*.parquet")
        if a.empty:
            continue
        idc = [c for c in a.columns if not re.search(r"(Amt|Cnt)$", c)]
        long = a.melt(id_vars=idc, var_name="metric", value_name="value")
        long["market"] = long.metric.str.replace(r"(Amt|Cnt)$", "", regex=True)
        long["measure"] = np.where(long.metric.str.endswith("Amt"), "price", "head_count")
        long = long.pivot_table(index=["ym", "gradeNm", "market"], columns="measure", values="value", aggfunc="first").reset_index()
        long["price"], long["head_count"] = num(long.get("price")), num(long.get("head_count"))
        frames.append(long.assign(species="소" if kind == "cattle_auction" else "돼지"))
    if frames:
        a = pd.concat(frames, ignore_index=True)
        a.insert(0, "month", pd.to_datetime(a.pop("ym"), format="%Y%m"))
        mk = pd.read_csv(RAW.parent / "reference" / "ekape_market.csv", dtype=str)[["market_cd", "market_nm", "region"]]
        a["market_cd"] = a.market.str.replace("c_", "", regex=False)
        a = a.merge(mk, on="market_cd", how="left").drop(columns="market")
        write(a.rename(columns={"gradeNm": "grade"}), "livestock", "auction_monthly")
    p = read_all("ekape/pig_rep_price/*.parquet")
    if not p.empty:
        write(pd.DataFrame({"date": pd.to_datetime(p.sumYmd, format="%Y%m%d"), "skin": p.sableGubn, "price": num(p.costAmt)})
              .drop_duplicates().sort_values("date"), "livestock", "pig_rep_price_daily")
    s = read_all("ekape/stock/*.parquet")
    if not s.empty:
        part = [c for c in s.columns if c.startswith("livestockPart_")] + ["totStock"]
        out = s[["ym", "judgeKindNm", "unit"] + part].copy()
        out[part] = out[part].apply(num)
        kg = out.unit.str.lower().eq("kg")  # 소는 kg, 돼지는 ton으로 제공 → ton으로 통일
        out.loc[kg, part] = out.loc[kg, part] / 1000
        out["unit"] = "ton"
        out.insert(0, "month", pd.to_datetime(out.pop("ym"), format="%Y%m"))
        write(out.rename(columns={"judgeKindNm": "species"}), "livestock", "stock_monthly")


def fishery():
    t = read_all("nifs/coast_temp/*.parquet")
    if not t.empty:
        write(pd.DataFrame({"date": pd.to_datetime(t.obs_dat, format="%Y%m%d"), "sea": t.gru_nam, "stn_cd": t.sta_cde, "stn_nm": t.sta_nam_kor,
                            "water_temp": num(t.wtr_tmp), "air_temp": num(t.air_tmp), "qc": t.qc_wtr}).sort_values(["stn_cd", "date"]),
              "fishery", "coast_temp_daily")
    r = read_all("nifs/redtide/*.parquet")
    if not r.empty:
        write(pd.DataFrame({"date": pd.to_datetime(r.day_report, format="%Y%m%d", errors="coerce"), "news_id": r.cod_news, "organism": r.nam_biology,
                            "area": r.txt_seas, "density_min": num(r.min_density), "density_max": num(r.max_density),
                            "water_temp_min": num(r.min_watertemp), "water_temp_max": num(r.max_watertemp)}), "fishery", "redtide_events")
    j = read_all("nifs/jellyfish/*.parquet")
    if not j.empty:
        write(pd.DataFrame({"date": pd.to_datetime(j.inpt_date, format="%Y%m%d"), "title": j.board_subject}), "fishery", "jellyfish_reports")


def buoy():
    """기상청 해양기상부이 시간자료(기상자료개방포털에서 수동 다운로드, data/raw/kma_buoy/*.csv) → 일평균 수온.
    해역 평균은 2016년부터 끊김 없이 운영된 부이 17곳(data/reference/kma_buoy_station.csv)만 사용해 구성 변화 편향을 막는다."""
    fs = sorted(glob.glob(str(RAW / "kma_buoy" / "*.csv")))
    if not fs:
        return
    h = pd.concat([pd.read_csv(f, encoding="cp949") for f in fs], ignore_index=True)
    h = h.rename(columns={"지점": "stn_id", "일시": "time", "수온(°C)": "water_temp", "풍속(m/s)": "wind"}).drop_duplicates(["stn_id", "time"])
    h["date"] = (pd.to_datetime(h.time) - pd.Timedelta(minutes=1)).dt.normalize()  # "24시(=다음날 00시)"를 전날로
    d = h.groupby(["stn_id", "date"]).agg(water_temp=("water_temp", "mean"), water_temp_max=("water_temp", "max"),
                                          wind=("wind", "mean"), n_hours=("water_temp", "count")).reset_index()
    d = d[d.date >= START]
    ref = pd.read_csv(RAW.parent / "reference" / "kma_buoy_station.csv")
    d = d.merge(ref, on="stn_id", how="left")
    d["core"] = d.stn_nm.notna()
    write(d, "fishery", "buoy_temp_daily")
    core = d[d.core & (d.n_hours >= 12)]  # 관측 12시간 미만인 날(연말 00시만 있는 날 등)은 제외
    sea = core.groupby(["date", "sea"]).agg(water_temp=("water_temp", "mean"), n_buoys=("stn_id", "nunique")).reset_index()
    write(sea, "fishery", "sea_temp_daily")


def recipe():
    b, i = read_all("recipe/basic.parquet"), read_all("recipe/ingredient.parquet")
    if b.empty:
        return
    write(b.drop(columns="ROW_NUM").rename(columns=str.lower), "common", "recipe_basic")
    write(i.drop(columns="ROW_NUM").rename(columns=str.lower), "common", "recipe_ingredient")


STEPS = [price, mafra_wholesale, at_trade, calendar, weather, ecos, customs, production, livestock, fishery, buoy, recipe]

if __name__ == "__main__":
    for step in STEPS:
        print(f"[{step.__name__}]")
        step()
