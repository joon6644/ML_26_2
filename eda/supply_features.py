"""공급 측 피처 (07 실험용). 모든 피처는 예측 시점 t에 공개돼 있던 값만 쓴다.

월·분기·연 통계는 '공개일'(기간 종료 + 공개 시차)을 만들어 merge_asof(backward)로 붙인다.
    공개 시차(보수적으로 잡음): 관세청 수입 = 월말 + 20일, 어업생산동향 = 월말 + 35일, 축평원 경락·재고 = 월말 + 20일·45일,
    가축동향 = 분기말 + 60일, 농작물 생산량 = 다음 해 3월 1일, 국제상품가격 = 월말 + 20일
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
REF = ROOT / "data" / "reference"
RAW = ROOT / "data" / "raw"
SHARE_YEARS = (2016, 2022)          # 주산지·주원산지 비중을 계산하는 기간 (학습 기간 안)


def _item_map():
    return pd.read_csv(REF / "item_map.csv", dtype=str).fillna("")


def _sido_alias():
    a = pd.read_csv(REF / "sido_alias.csv", dtype=str)
    return dict(zip(a.name, a.sido))


def month_end(p):
    return (pd.to_datetime(p) + pd.offsets.MonthEnd(0)).normalize()


def asof(df, feat, by, cols):
    """feat: [by..., 'avail', cols...]. 각 행의 date 시점에 공개돼 있던 가장 최근 값을 붙인다."""
    if feat.empty:
        return df.assign(**{c: np.nan for c in cols})
    feat = feat.dropna(subset=["avail"]).sort_values("avail")
    left = df.reset_index().sort_values("date")
    m = pd.merge_asof(left, feat[by + ["avail"] + cols].rename(columns={"avail": "date"}), on="date", by=by, direction="backward")
    return m.set_index("index").sort_index()


def log_yoy(s, lag):
    return np.log(s / s.shift(lag))


# ------------------------------------------------------------------ 공통: 품목별 HS 수입량

def imports_feature(df):
    """item_map.hs_prefixes 로 품목별 월 수입 중량을 합산 → 전년 동월 대비, 3개월 합 전년 대비."""
    imp = pd.concat([pd.read_parquet(P / d / "import_monthly.parquet") for d in ("agri", "livestock", "fishery")], ignore_index=True)
    rows = []
    for r in _item_map().itertuples():
        pre = tuple(x for x in r.hs_prefixes.split(";") if x)
        if not pre:
            continue
        m = imp[imp.hs_code.str.startswith(pre)].groupby("month").imp_kg.sum()
        m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="MS")) if len(m) else m
        if m.empty:
            continue
        rows.append(pd.DataFrame({"ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd, "avail": month_end(m.index) + pd.Timedelta(days=20),
                                  "imp_yoy": log_yoy(m + 1, 12).values, "imp3_yoy": log_yoy(m.rolling(3).sum() + 1, 12).values}))
    return asof(df, pd.concat(rows), ["ctgry_cd", "item_cd"], ["imp_yoy", "imp3_yoy"])


# ------------------------------------------------------------------ 공통: 같은 품목의 도매(중도매) 가격 신호

def wholesale_signal(df, daily):
    """품목별 중도매 시계열들을 각자 평균으로 정규화해 평균낸 지수 → 7·28일 변화. 소매·도매 행 모두에 붙인다."""
    w = daily[daily.se_nm == "중도매"].copy()
    if w.empty:
        return df.assign(whsl_r7=np.nan, whsl_r28=np.nan)
    w["lp"] = np.log(w.price_kg)
    w["lp"] -= w.groupby(["ctgry_cd", "item_cd", "vrty_nm", "grd_nm"]).lp.transform("mean")
    idx = w.groupby(["ctgry_cd", "item_cd", "date"]).lp.mean().reset_index()
    parts = []
    for (c, i), g in idx.groupby(["ctgry_cd", "item_cd"]):
        s = g.set_index("date").lp.asfreq("D")
        m7 = s.rolling(7, min_periods=3).mean()
        parts.append(pd.DataFrame({"date": s.index, "ctgry_cd": c, "item_cd": i,
                                   "whsl_r7": (m7 - m7.shift(7)).values, "whsl_r28": (m7 - m7.shift(28)).values}))
    return df.merge(pd.concat(parts), on=["date", "ctgry_cd", "item_cd"], how="left")


# ------------------------------------------------------------------ 농산

def garak_vs_normal(df):
    """가락 kg 물량의 14·30일 합을 과거 1~3년 같은 창 평균과 비교 (log)."""
    t = pd.read_parquet(P / "agri" / "trade_by_item_daily.parquet")
    parts = []
    for (c, i), g in t.groupby(["ctgry_cd", "item_cd"]):
        q = g.set_index("date").qty_kg.asfreq("D").fillna(0)
        q = q[q.index >= "2018-01-03"]
        out = {"date": q.index, "ctgry_cd": c, "item_cd": i}
        for w in (14, 30):
            s = q.rolling(w).sum()
            norm = pd.concat([s.shift(364 * k) for k in (1, 2, 3)], axis=1).mean(axis=1)    # 과거 1~3년 같은 시기 평균
            out[f"garak_q{w}_vs_norm"] = np.log((s + 1) / (norm + 1)).values
        parts.append(pd.DataFrame(out))
    return df.merge(pd.concat(parts), on=["date", "ctgry_cd", "item_cd"], how="left")


def origin_daily():
    """가락 정산 원본에서 일 × 가락품목 × 원산지 시도별 kg 물량 (캐시)."""
    cache = P / "agri" / "trade_origin_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    alias = _sido_alias()
    parts = []
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(str(RAW / "at_trade" / "*" / "*.parquet"))):
        if pq.ParquetFile(f).metadata.num_rows == 0:                                   # 휴장일은 컬럼 없는 빈 파일
            continue
        d = pd.read_parquet(f, columns=["trd_clcln_ymd", "gds_lclsf_nm", "gds_mclsf_nm", "plor_nm", "unit_nm", "unit_tot_qty"])
        d = d[d.unit_nm == "kg"].copy()
        if d.empty:
            continue
        first = [str(v).split()[0] if isinstance(v, str) and v.strip() else "" for v in d.plor_nm]    # 원산지의 첫 단어 = 시도
        d["sido"] = pd.Series(first, index=d.index).map(alias).replace({"전남+광주": "전남"}).fillna("기타")                      # 시도가 아니면(수입국 등) 기타
        d["qty"] = pd.to_numeric(d.unit_tot_qty, errors="coerce")
        parts.append(d.groupby(["trd_clcln_ymd", "gds_lclsf_nm", "gds_mclsf_nm", "sido"]).qty.sum().reset_index())
    o = pd.concat(parts, ignore_index=True)
    o["date"] = pd.to_datetime(o.pop("trd_clcln_ymd"))
    o["garak_key"] = o.gds_lclsf_nm + ">" + o.gds_mclsf_nm
    o = o[["date", "garak_key", "sido", "qty"]]
    o.to_parquet(cache, index=False)
    return o


def origin_share(df):
    """품목의 '주 원산지'(SHARE_YEARS 동안 물량 1위 시도)의 최근 14일 물량 비중 − 과거 1~3년 같은 시기 비중."""
    o = origin_daily()
    link = [(r.ctgry_cd, r.item_cd, k) for r in _item_map().itertuples() for k in r.garak_item.split(";") if k]
    parts = []
    for c, i, k in link:
        g = o[o.garak_key == k]
        if g.empty:
            continue
        base = g[(g.date.dt.year >= SHARE_YEARS[0]) & (g.date.dt.year <= SHARE_YEARS[1])]
        top = base[base.sido != "기타"].groupby("sido").qty.sum().idxmax() if len(base[base.sido != "기타"]) else None
        if top is None:
            continue
        tot = g.groupby("date").qty.sum().asfreq("D").fillna(0)
        topq = g[g.sido == top].groupby("date").qty.sum().reindex(tot.index).fillna(0)
        share = topq.rolling(14).sum() / tot.rolling(14).sum().replace(0, np.nan)
        norm = pd.concat([share.shift(364 * j) for j in (1, 2, 3)], axis=1).mean(axis=1)
        parts.append(pd.DataFrame({"date": share.index, "ctgry_cd": c, "item_cd": i, "origin_top_share_dev": (share - norm).values}))
    f = pd.concat(parts).groupby(["date", "ctgry_cd", "item_cd"]).origin_top_share_dev.mean().reset_index()   # 가락 품목이 둘 이상이면 평균
    return df.merge(f, on=["date", "ctgry_cd", "item_cd"], how="left")


def production_weather(df):
    """품목별 주산지 기상: SHARE_YEARS의 시도별 생산량 비중 × 시도별 일 기상 → 최근 30일 폭우·폭염·한파 일수, 14일 기온 편차."""
    alias = _sido_alias()
    prod = pd.read_parquet(P / "agri" / "production_annual.parquet")
    prod = prod[prod.item_nm.str.endswith(":생산량")].assign(sido=lambda d: d["시도별"].map(alias), year=lambda d: d.period.astype(int))
    prod = prod[prod.sido.notna() & ~prod.sido.isin(["전국"]) & prod.year.between(*SHARE_YEARS)]
    st = pd.read_parquet(P / "common" / "weather_station_daily.parquet")
    st = st.merge(pd.read_csv(REF / "asos_station_sido.csv")[["stnId", "sido"]].rename(columns={"stnId": "stn_id"}).astype({"stn_id": str}),
                  on="stn_id", how="left")
    by_sido = st.groupby(["sido", "date"])[["rain", "temp_max", "temp_min", "temp_avg"]].mean()
    piv = {c: by_sido[c].unstack("sido").asfreq("D") for c in ["rain", "temp_max", "temp_min", "temp_avg"]}   # 날짜 × 시도
    parts = []
    for r in _item_map().itertuples():
        if not r.kosis_production or r.kosis_production.split(":")[0] not in ("veg_leaf", "veg_root", "veg_seasoning", "fruit", "food_crops"):
            continue
        tbl, name = r.kosis_production.split(":")
        p = prod[(prod.table == tbl) & (prod.item_nm == f"{name}:생산량")]
        if p.empty:
            continue
        w = p.assign(sido=p.sido.replace({"전남+광주": "전남"})).groupby("sido").value.sum()
        w = w[(w > 0) & w.index.isin(piv["rain"].columns)]
        w = w / w.sum()
        ws = pd.DataFrame({c: (v[w.index] * w).sum(axis=1, min_count=1) / (v[w.index].notna() * w).sum(axis=1).replace(0, np.nan)
                           for c, v in piv.items()})                                                    # 결측 관측소는 빼고 가중평균
        clim = ws.temp_avg[(ws.index.year >= SHARE_YEARS[0]) & (ws.index.year <= SHARE_YEARS[1])]
        clim = clim.groupby(clim.index.month).mean()
        parts.append(pd.DataFrame({
            "date": ws.index, "ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd,
            "pw_heavy30": (ws.rain >= 30).rolling(30, min_periods=20).sum().values,       # 주산지 가중 일강수 30mm 이상 일수
            "pw_heat30": (ws.temp_max >= 33).rolling(30, min_periods=20).sum().values,
            "pw_cold30": (ws.temp_min <= -10).rolling(30, min_periods=20).sum().values,
            "pw_rain30": ws.rain.rolling(30, min_periods=20).sum().values,
            "pw_temp14_anom": (ws.temp_avg.rolling(14, min_periods=10).mean() - ws.index.month.map(clim).values).values}))
    return df.merge(pd.concat(parts), on=["date", "ctgry_cd", "item_cd"], how="left")


def production_yoy(df):
    """전년 생산량(전국) 증감. Y년 값은 Y+1년 3월 1일부터 공개된 것으로 본다."""
    prod = pd.read_parquet(P / "agri" / "production_annual.parquet")
    nat = prod[prod["시도별"].isin(["계", "전국"]) & prod.item_nm.str.endswith(":생산량")]
    rows = []
    for r in _item_map().itertuples():
        if ":" not in r.kosis_production or r.domain != "agri":
            continue
        tbl, name = r.kosis_production.split(":")
        s = nat[(nat.table == tbl) & (nat.item_nm == f"{name}:생산량")].groupby("period").value.sum().sort_index()
        if len(s) < 2:
            continue
        s.index = s.index.astype(int)
        rows.append(pd.DataFrame({"ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd,
                                  "avail": pd.to_datetime([f"{y + 1}-03-01" for y in s.index]), "prod_yoy": np.log(s / s.shift(1)).values}))
    return asof(df, pd.concat(rows), ["ctgry_cd", "item_cd"], ["prod_yoy"])


def agri_supply(df, daily):
    for fn in (garak_vs_normal, origin_share, production_weather, production_yoy, imports_feature):
        df = fn(df)
    return wholesale_signal(df, daily)


AGRI_COLS = ["garak_q14_vs_norm", "garak_q30_vs_norm", "origin_top_share_dev", "pw_heavy30", "pw_heat30", "pw_cold30", "pw_rain30",
             "pw_temp14_anom", "prod_yoy", "imp_yoy", "imp3_yoy", "whsl_r7", "whsl_r28"]


# ------------------------------------------------------------------ 축산

CENSUS = {"소": ("livestock_census", "한우:마리수"), "돼지": ("pig", "마리수"), "계란": ("chicken_census", "산란계:마리수"),
          "닭": ("chicken_census", "육용계:마리수"), "우유": ("livestock_census", "젖소:마리수")}
SPECIES = {"소": "소", "수입 소고기": "소", "돼지": "돼지", "수입 돼지고기": "돼지"}
DISEASE = {"계란": "고병원성조류인플루엔자", "닭": "고병원성조류인플루엔자", "돼지": "아프리카돼지열병",
           "수입 돼지고기": "아프리카돼지열병", "소": "구제역", "수입 소고기": "구제역", "우유": "구제역"}


def livestock_supply(df, daily):
    ce = pd.read_parquet(P / "livestock" / "census_quarterly.parquet")
    ce = ce[(ce["시도별"] == "전국") & (ce["사육규모별"].fillna("합계") == "합계")]
    rows = []
    for item, (tbl, nm) in CENSUS.items():
        if tbl == "pig":
            s = ce[ce.table.isin(["pig_census_old", "pig_census"]) & (ce.item_nm == nm)]
            s = s.sort_values("table").drop_duplicates("period", keep="last")      # 2017년 겹치는 분기는 새 통계표 사용
        else:
            s = ce[(ce.table == tbl) & (ce.item_nm == nm)]
        s = s.groupby("period").value.sum().sort_index()
        q_end = pd.PeriodIndex([f"{p[:4]}Q{int(p[4:])}" for p in s.index], freq="Q").end_time.normalize()
        rows.append(pd.DataFrame({"item_nm": item, "avail": q_end + pd.Timedelta(days=60),
                                  "census_yoy": np.log(s / s.shift(4)).values, "census_qoq": np.log(s / s.shift(1)).values}))
    df = asof(df, pd.concat(rows), ["item_nm"], ["census_yoy", "census_qoq"])

    d = pd.read_parquet(P / "livestock" / "disease_events.parquet")
    idx = pd.date_range("2015-01-01", "2026-09-30")
    heads = {k: d[d.disease_std == k].groupby("date").head_count.sum().reindex(idx, fill_value=0) for k in set(DISEASE.values())}
    for w in (28, 90):                                                                  # 최근 w일 발생 두수 합 (log1p)
        roll = {k: np.log1p(v.rolling(w).sum()) for k, v in heads.items()}
        df[f"cull{w}"] = [roll[DISEASE[i]].get(t, np.nan) for i, t in zip(df.item_nm, df.date)]

    a = pd.read_parquet(P / "livestock" / "auction_monthly.parquet")
    a = a[(a.market_cd == "CTot") & (a.grade == "평균")].sort_values("month")
    rows = []
    for sp, g in a.groupby("species"):
        g = g.set_index("month")
        for item, s2 in SPECIES.items():
            if s2 == sp:
                rows.append(pd.DataFrame({"item_nm": item, "avail": month_end(g.index) + pd.Timedelta(days=20),
                                          "auction_heads_yoy": log_yoy(g.head_count, 12).values, "auction_price_m1": log_yoy(g.price, 1).values}))
    df = asof(df, pd.concat(rows), ["item_nm"], ["auction_heads_yoy", "auction_price_m1"])

    s = pd.read_parquet(P / "livestock" / "stock_monthly.parquet").sort_values("month")
    rows = []
    for sp, g in s.groupby("species"):
        g = g.set_index("month")
        for item, s2 in SPECIES.items():
            if s2 == sp:
                rows.append(pd.DataFrame({"item_nm": item, "avail": month_end(g.index) + pd.Timedelta(days=45), "stock_yoy": log_yoy(g.totStock, 12).values}))
    df = asof(df, pd.concat(rows), ["item_nm"], ["stock_yoy"])

    m = pd.read_parquet(P / "common" / "macro_monthly.parquet")
    feed = m[(m.table == "intl_commodity") & m.name.isin(["옥수수", "대두"])].pivot_table(index="month", columns="name", values="value").mean(axis=1)
    feed = pd.DataFrame({"k": 1, "avail": month_end(feed.index) + pd.Timedelta(days=20), "feed_6m": log_yoy(feed, 6).values})
    df = asof(df.assign(k=1), feed, ["k"], ["feed_6m"]).drop(columns="k")
    df = imports_feature(df)
    return wholesale_signal(df, daily)


LIVE_COLS = ["census_yoy", "census_qoq", "cull28", "cull90", "auction_heads_yoy", "auction_price_m1", "stock_yoy", "feed_6m",
             "imp_yoy", "imp3_yoy", "whsl_r7", "whsl_r28"]


# ------------------------------------------------------------------ 수산

def fishery_supply(df, daily):
    fp = pd.read_parquet(P / "fishery" / "production_monthly.parquet")
    fp = fp[(fp["어업별"] == "계") & (fp.item_id == "T01")]
    rows = []
    for r in _item_map().itertuples():
        if not r.kosis_production.startswith("fishery:"):
            continue
        name = r.kosis_production.split(":", 1)[1]
        s = fp[fp["품종별"] == name].groupby("period").value.sum().sort_index()
        if len(s) < 13:
            continue
        mi = pd.to_datetime(s.index, format="%Y%m")
        rows.append(pd.DataFrame({"ctgry_cd": r.ctgry_cd, "item_cd": r.item_cd, "avail": month_end(mi) + pd.Timedelta(days=35),
                                  "catch_yoy": log_yoy(s + 1, 12).values, "catch3_yoy": log_yoy(s.rolling(3).sum() + 1, 12).values}))
    df = asof(df, pd.concat(rows), ["ctgry_cd", "item_cd"], ["catch_yoy", "catch3_yoy"])
    df = imports_feature(df)

    rt = pd.read_parquet(P / "fishery" / "redtide_events.parquet")
    idx = pd.date_range("2015-01-01", "2026-09-30")
    cnt = rt.dropna(subset=["date"]).groupby("date").news_id.nunique().reindex(idx, fill_value=0).rolling(28).sum()
    df["redtide28"] = df.date.map(cnt)

    sea = pd.read_parquet(P / "fishery" / "sea_temp_daily.parquet").pivot(index="date", columns="sea", values="water_temp").asfreq("D")
    tr = sea[(sea.index.year >= SHARE_YEARS[0]) & (sea.index.year <= SHARE_YEARS[1])]
    clim = tr.groupby(tr.index.month).mean()
    s14 = sea.rolling(14, min_periods=7).mean()
    anom = s14 - clim.reindex(s14.index.month).values
    anom.columns = [f"sea14_anom_{c}" for c in anom.columns]
    df = df.merge(anom.rename_axis("date").reset_index(), on="date", how="left")
    return wholesale_signal(df, daily)


FISH_COLS = ["catch_yoy", "catch3_yoy", "imp_yoy", "imp3_yoy", "redtide28", "sea14_anom_남해", "sea14_anom_동해", "sea14_anom_서해", "whsl_r7", "whsl_r28"]

SUPPLY = {"agri": (agri_supply, AGRI_COLS), "livestock": (livestock_supply, LIVE_COLS), "fishery": (fishery_supply, FISH_COLS)}
