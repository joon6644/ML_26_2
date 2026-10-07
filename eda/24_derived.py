# %% [markdown]
# # 파생변수 · 모델 실험 공용 도구 (기본 피처 = data/reference/feature_selection_base_v1.csv 고정)
# 통합 모델(22번 구성) 위에 파생변수를 붙이고, 같은 시드 비교로 효과를 잰다. 가설·결과 기록: eda/derived_log.md
# 지표: RMSE, MAE, R², 방향 정확도(|y|>1% 행에서 부호 일치). 2026-10-07 이전 기록은 ±2%·3구간·상승재현율 기준

# %%
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cs", ROOT / "eda" / "22_combined_selection.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
BASE = pd.read_csv(ROOT / "data" / "reference" / "feature_selection_base_v1.csv")
BAND = 0.01   # 2026-10-07 확정: 방향 정확도·앱 표시·신뢰도 모두 ±1% (이전 ±2%·3구간·상승재현율은 폐기)


def metrics(y, p):
    """RMSE·MAE·R² + 방향 정확도 (실제 변화 |y|>1% 행에서 예측과 실제의 부호 일치)."""
    m = np.abs(y) > BAND
    return {"RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MAE": float(np.mean(np.abs(p - y))),
            "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "방향정확도": float(np.mean(np.sign(p[m]) == np.sign(y[m])))}


def base_sets(nm):
    k = BASE[BASE.분야 == nm].feature.tolist()
    cats = [c for c in cs.fs.bf.CAT + ["sgg_nm"] if c in k]
    return [c for c in k if c not in cats], cats


def masks(df):
    t0, t1, s0, s1 = cs.SPLIT
    return ((df.date >= t0) & (df.date <= t1)).values, ((df.date >= s0) & (df.date <= s1)).values


# %% 파생변수 1차 (가설: eda/derived_log.md D1~D6)
SKEY = cs.KEYS + ["sgg_nm"]


def _prior_year_mean(t, keys, col, min_years=2):
    """keys×주차×연도 평균 → 그 해보다 이전 연도들의 평균 (누수 없음)."""
    g = t.groupby(keys + ["woy", "year"])[col].mean().rename("v").reset_index().sort_values("year")
    grp = g.groupby(keys + ["woy"])
    g["n"] = grp.cumcount()
    g["prior"] = grp.v.transform(lambda s: s.shift(1).expanding().mean())
    g.loc[g.n < min_years, "prior"] = np.nan
    return g[keys + ["woy", "year", "prior"]]


def _rolling_mean(df, col, window, key=SKEY, min_periods=None):
    s = df.sort_values(key + ["date"])
    mp = min_periods or max(3, int(pd.Timedelta(window).days * 0.3))   # 창 길이의 30% 이상 관측
    r = s.groupby(key, observed=True, sort=False).rolling(window, on="date", min_periods=mp)[col].mean()
    return pd.Series(r.values, index=s.index).reindex(df.index)


def add_derived(df, d):
    scope = dict(df.attrs.get("scope", {}))   # merge가 attrs를 지우므로 먼저 보관 (행 순서는 left merge로 유지)
    n0 = len(df)
    df = df.copy()
    df["year"] = df.date.dt.year
    # D1: 같은 주 평년 변화율 (전국 시계열, 이전 연도만)
    nat = cs.fs.frame(d)[cs.KEYS + ["date", "y", "base"]]
    nat["woy"], nat["year"] = nat.date.dt.isocalendar().week.astype(int), nat.date.dt.year
    clim = _prior_year_mean(nat, cs.KEYS, "y").rename(columns={"prior": "clim_y_woy"})
    df = df.merge(clim, on=cs.KEYS + ["woy", "year"], how="left")
    # D2: 최근 1년 평균 대비 가격 (자기 시계열)
    df["lvl_52w"] = np.log(df.base / _rolling_mean(df, "base", "365D"))
    # D3: 평년 같은 주의 1년 평균 대비 위치와의 차이 (전국 시계열의 이전 연도 평균)
    nat = nat.sort_values(cs.KEYS + ["date"])
    nat["sgg_nm"] = "전국"
    nat["lvl"] = np.log(nat.base / _rolling_mean(nat, "base", "365D"))
    norm = _prior_year_mean(nat, cs.KEYS, "lvl").rename(columns={"prior": "lvl_norm"})
    df = df.merge(norm, on=cs.KEYS + ["woy", "year"], how="left")
    df["lvl_vs_season"] = df.lvl_52w - df.lvl_norm
    # D4·D5: 도매가 (농: 가락, 수: aT 중도매)
    w = {"agri": "garak_price_kg", "fishery": "whsl_price_kg"}.get(d)
    if w and w in df:
        gap = np.log(df[w] / df.base).replace([np.inf, -np.inf], np.nan)
        df["_gap"] = gap
        df["whsl_gap_anom"] = gap - _rolling_mean(df, "_gap", "91D")
        df["whsl_mom28"] = np.log(df[w] / _rolling_mean(df, w, "28D")).replace([np.inf, -np.inf], np.nan)
        df = df.drop(columns="_gap")
    # D6: 최근 2주 호우·폭염·한파 (농, 전국 일기상)
    if d == "agri":
        wx = pd.read_parquet(cs.P / "common" / "weather_national_daily.parquet", columns=["date", "rain", "temp_max", "temp_min"]).set_index("date").asfreq("D")
        e = pd.DataFrame({"wx_rain_14d": wx.rain.rolling(14, min_periods=7).sum(),
                          "wx_heat_14d": (wx.temp_max >= 33).rolling(14, min_periods=7).sum(),
                          "wx_cold_14d": (wx.temp_min <= -5).rolling(14, min_periods=7).sum()}).reset_index()
        df = df.merge(e, on="date", how="left")
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df


DERIVED = {"agri": ["clim_y_woy", "lvl_52w", "lvl_vs_season", "whsl_gap_anom", "whsl_mom28", "wx_rain_14d", "wx_heat_14d", "wx_cold_14d"],
           "livestock": ["clim_y_woy", "lvl_52w", "lvl_vs_season"],
           "fishery": ["clim_y_woy", "lvl_52w", "lvl_vs_season", "whsl_gap_anom", "whsl_mom28"]}


# %% 파생변수 2차 (가설: eda/derived_log.md D5′ D7 D8 D11 D13 D14)
def _vs_season(df, d, col, out, roll=None):
    """col(또는 그 roll 평균)의 log(값 / 이전 연도 같은 주 평균). 전국 시계열 기준으로 평년을 만들고 KEYS·주차·연도로 붙임."""
    nat = cs.fs.frame(d)[cs.KEYS + ["date", col]].copy()
    nat["sgg_nm"] = "전국"
    if roll:
        nat[col] = _rolling_mean(nat, col, roll, min_periods=3)
    nat["woy"], nat["year"] = nat.date.dt.isocalendar().week.astype(int), nat.date.dt.year
    nat["lv"] = np.log(nat[col].where(nat[col] > 0))
    norm = _prior_year_mean(nat, cs.KEYS, "lv").rename(columns={"prior": "_norm"})
    cur = nat[cs.KEYS + ["date", "lv"]].rename(columns={"lv": "_cur"})
    df = df.merge(cur, on=cs.KEYS + ["date"], how="left").merge(norm, on=cs.KEYS + ["woy", "year"], how="left")
    df[out] = df._cur - df._norm
    return df.drop(columns=["_cur", "_norm"])


def add_derived2(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    if "year" not in df:
        df["year"] = df.date.dt.year
    w = {"agri": "garak_price_kg", "fishery": "whsl_price_kg"}.get(d)
    if w and w in df:
        for k in (7, 28):
            df[f"whsl_mom{k}"] = np.log(df[w] / _rolling_mean(df, w, f"{k}D")).replace([np.inf, -np.inf], np.nan)
    if d == "agri":
        df = _vs_season(df, d, "garak_qty_kg", "qty_vs_season", roll="7D")
    if d == "livestock":
        pork = df.item_nm.isin(["돼지", "수입 돼지고기"]).values
        df["_g"] = np.log(df.pig_auction_price_kg / df.base).where(pork)
        df["pig_gap_anom"] = df._g - _rolling_mean(df, "_g", "91D")
        df = df.drop(columns="_g")
        scope["pig_gap_anom"] = pork
        df = _vs_season(df, d, "auction_heads", "heads_vs_season")
    if d == "fishery":
        df = _vs_season(df, d, "catch_ton", "catch_vs_season")
    df["premium_mom28"] = df.premium - _rolling_mean(df, "premium", "28D")
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df


DERIVED2 = {"agri": ["whsl_mom7", "whsl_mom28", "qty_vs_season", "premium_mom28"],
            "livestock": ["pig_gap_anom", "heads_vs_season", "premium_mom28"],
            "fishery": ["whsl_mom7", "whsl_mom28", "catch_vs_season", "premium_mom28"]}
ADOPTED = {"agri": ["whsl_gap_anom", "wx_rain_14d", "wx_sun_14d_anom", "wx_temp_14d_anom"], "livestock": [], "fishery": []}   # 채택 누적


# %% 파생변수 3차 (D15·D16·D17)
def add_derived3(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    df["premium_norm"] = _rolling_mean(df, "premium", "365D")
    df["premium_anom"] = df.premium - df.premium_norm
    s = df.sort_values(SKEY + ["date"])
    chg = s.groupby(SKEY, observed=True, sort=False).price_kg.diff().fillna(0).ne(0)
    last = s.date.where(chg).groupby([s[k] for k in SKEY], observed=True, sort=False).ffill()
    df["days_since_change"] = ((s.date - last).dt.days).reindex(df.index)
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df


DERIVED3 = ["premium_anom", "premium_norm", "days_since_change"]


# %% 분야별 학습 설정 (4차 결과로 채택) + 공용 학습 함수
TRAIN_CFG = {"agri": {"clip": None, "params": {}},
             "livestock": {"clip": (0.01, 0.99), "params": {}},
             "fishery": {"clip": None, "params": {"objective": "huber", "alpha": 0.05}}}
LGB_BASE = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=0, verbose=-1)


def matrix(df, num, cats, tr, seed=0):
    num = sorted(num)
    num = [num[i] for i in np.random.default_rng(seed).permutation(len(num))]   # 피처 순서도 시드로
    X = cs.rm.prep(df, num, tr)
    for c, ok in df.attrs.get("scope", {}).items():
        if c in X:
            X.loc[~ok, c] = -1.0
    for c in cats:
        X[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    return X


def train_predict(df, d, num, cats, tr, te, seed=0, cfg=None, target=None, n_estimators=None):
    import lightgbm as lgb
    cfg = cfg or TRAIN_CFG[d]
    X = matrix(df, num, cats, tr, seed)
    y = (df.y.values if target is None else target)[tr]
    if cfg.get("clip"):
        y = np.clip(y, *np.nanquantile(y, cfg["clip"]))
    p = {**LGB_BASE, **cfg.get("params", {}), "random_state": seed}
    if cfg.get("mono"):
        p["monotone_constraints"] = [MONO[d].get(c, 0) for c in X.columns]
        p["monotone_constraints_method"] = "advanced"
    if n_estimators:
        p["n_estimators"] = n_estimators
    m = lgb.LGBMRegressor(**p).fit(X[tr], y, categorical_feature=cats)
    return m.predict(X[te]), m, X


# %% 파생변수 4차 (D18·D19)
def add_derived4(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    if d == "agri":
        # D18: 주산지 가중 일기상(prodarea_*)을 시계열별 최근 14일 누적
        s = df[cs.KEYS + ["date", "prodarea_rain", "prodarea_temp_max", "prodarea_temp_min"]].drop_duplicates(cs.KEYS + ["date"]).copy()
        s["sgg_nm"] = "전국"
        s["_heat"] = (s.prodarea_temp_max >= 33).astype(float).where(s.prodarea_temp_max.notna())
        s["_cold"] = (s.prodarea_temp_min <= -5).astype(float).where(s.prodarea_temp_min.notna())
        for c, out in [("prodarea_rain", "prod_rain_14d"), ("_heat", "prod_heat_14d"), ("_cold", "prod_cold_14d")]:
            s[out] = _rolling_mean(s, c, "14D", min_periods=5) * 14
        df = df.merge(s[cs.KEYS + ["date", "prod_rain_14d", "prod_heat_14d", "prod_cold_14d"]], on=cs.KEYS + ["date"], how="left")
    # D19: 같은 계열 다른 시계열의 오늘 튐 (자기 제외, 같은 날·같은 지역 구분 없이 전국 행 기준)
    nat = df[df.is_nat == 1]
    g = nat.groupby(["date", "food_group"]).dev_last
    s_, n_ = g.transform("sum"), g.transform("count")
    v = (s_ - nat.dev_last.fillna(0)) / (n_ - nat.dev_last.notna().astype(int)).replace(0, np.nan)
    m = nat[cs.KEYS + ["date"]].assign(grp_dev_last=v.values)
    df = df.merge(m, on=cs.KEYS + ["date"], how="left")
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df


DERIVED4 = {"agri": ["prod_rain_14d", "prod_heat_14d", "prod_cold_14d", "grp_dev_last"], "livestock": ["grp_dev_last"], "fishery": ["grp_dev_last"]}


# %% 파생변수 5차 (D20·D21): 전국 일기상 2주 누적의 평년 같은 주 대비 (이전 연도만)
def add_derived5(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    wx = pd.read_parquet(cs.P / "common" / "weather_national_daily.parquet", columns=["date", "sunshine_hr", "temp_avg"]).set_index("date").asfreq("D")
    e = pd.DataFrame({"sun14": wx.sunshine_hr.rolling(14, min_periods=10).sum(), "t14": wx.temp_avg.rolling(14, min_periods=10).mean()}).reset_index()
    e["woy"], e["year"] = e.date.dt.isocalendar().week.astype(int), e.date.dt.year
    for c, out in [("sun14", "wx_sun_14d_anom"), ("t14", "wx_temp_14d_anom")]:
        yw = e.groupby(["woy", "year"])[c].mean().rename("v").reset_index().sort_values("year")
        yw["prior"] = yw.groupby("woy").v.transform(lambda s: s.shift(1).expanding().mean())
        e = e.merge(yw[["woy", "year", "prior"]], on=["woy", "year"], how="left")
        e[out] = e[c] - e.prior
        e = e.drop(columns="prior")
    df = df.merge(e[["date", "wx_sun_14d_anom", "wx_temp_14d_anom"]], on="date", how="left")
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df



def build_all(d):
    """통합 모델 데이터 + 채택 파생변수 전부 (1차·5차 함수)."""
    df = add_derived(cs.build(d)[0], d)
    if d == "agri":
        df = add_derived5(df, d)
    return df


ENSEMBLE = {"agri": "lgb5+xgb+cat", "livestock": "lgb5", "fishery": "lgb5"}   # 6차 채택, 축산물은 13차(단조 제약) 후 LGB5로: 앙상블과 차이 0.0001(잡음) — 제약 없는 XGB·Cat을 섞으면 가설 방향이 흐려짐


# %% 파생변수 6차 (D24·D25): 한우 주간 경락가 (data/raw/ekape/cattle_auction_weekly)
def cattle_weekly():
    import glob
    a = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ROOT / "data" / "raw" / "ekape" / "cattle_auction_weekly" / "*.parquet"))], ignore_index=True)
    a = a[a.gradeNm == "평균"].copy()
    a["price"] = pd.to_numeric(a.CTotAmt, errors="coerce")
    a["week_start"] = pd.to_datetime(a.week_start, format="%Y%m%d")
    a = a.sort_values("week_start")[["week_start", "price"]].dropna()
    a["avail"] = (a.week_start + pd.Timedelta(days=7)).astype("datetime64[ns]")   # 일요일까지 집계 → 다음 월요일부터 사용
    a["cattle_w_mom4"] = np.log(a.price / a.price.shift(1).rolling(4, min_periods=3).mean())
    return a


def add_derived6(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    if d == "livestock":
        a = cattle_weekly()
        order = np.argsort(df.date.values, kind="stable")
        left = df.iloc[order][["date"]].reset_index(); left["date"] = left.date.astype("datetime64[ns]")
        m = pd.merge_asof(left, a[["avail", "price", "cattle_w_mom4"]].rename(columns={"avail": "date", "price": "_cw"}), on="date", direction="backward")
        m = m.set_index("index").reindex(df.index)
        beef = df.item_nm.isin(["소", "수입 소고기"]).values
        df["cattle_w_mom4"] = m.cattle_w_mom4.where(beef).values
        df["_g"] = np.log(m._cw.values / df.base).where(beef)
        df["cattle_w_gap_anom"] = df._g - _rolling_mean(df, "_g", "91D")
        df = df.drop(columns="_g")
        scope["cattle_w_mom4"] = beef; scope["cattle_w_gap_anom"] = beef
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df



# %% 파생변수 7차 (D27)
def add_derived7(df, d):
    scope = dict(df.attrs.get("scope", {}))
    if d == "agri" and "whsl_price_kg" in df:
        df["_g"] = np.log(df.whsl_price_kg / df.base).replace([np.inf, -np.inf], np.nan)
        df["at_gap_anom"] = df._g - _rolling_mean(df, "_g", "91D")
        df = df.drop(columns="_g")
    df.attrs["scope"] = scope
    return df


# %% 파생변수 8차 (D28·D29·D30): 지역 가격의 시장 구성 효과 (price.parquet 시장 단위)
def market_features(d):
    RK = cs.KEYS + ["sgg_nm"]
    p = pd.read_parquet(cs.P / d / "price.parquet", columns=["date", "se_nm", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "sgg_nm", "mrkt_cd", "price_kg"])
    p = p[p.se_nm.isin(["소매", "중도매"]) & (p.price_kg > 0) & (p.sgg_nm != "전국") & (p.date >= "2015-01-01")]
    p = p.groupby(RK + ["mrkt_cd", "date"], observed=True).price_kg.median().reset_index().sort_values(RK + ["mrkt_cd", "date"])
    r = p.groupby(RK + ["mrkt_cd"], observed=True, sort=False).rolling("7D", on="date", min_periods=1).price_kg.mean()
    p["mdev"] = np.log(p.price_kg.values / r.values)
    p["lp"] = np.log(p.price_kg)
    g = p.groupby(RK + ["date"], observed=True).agg(n_mkt=("mdev", "size"), same_mkt_dev=("mdev", "mean"), mkt_disp=("lp", "std")).reset_index()
    g = g.sort_values(RK + ["date"])
    rn = g.groupby(RK, observed=True, sort=False).rolling("28D", on="date", min_periods=3).n_mkt.mean()
    g["n_mkt_dev"] = np.log(g.n_mkt.values / rn.values)
    return g[RK + ["date", "same_mkt_dev", "mkt_disp", "n_mkt_dev"]]


def add_derived8(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    g = market_features(d)
    for c in cs.KEYS + ["sgg_nm"]:
        if str(df[c].dtype) != str(g[c].dtype):
            g[c] = g[c].astype(df[c].dtype)
    df = df.merge(g, on=cs.KEYS + ["sgg_nm", "date"], how="left")
    reg = df.is_nat.values == 0
    df["comp_effect"] = (df.dev_last - df.same_mkt_dev).where(reg)
    for c in ["n_mkt_dev", "mkt_disp"]:
        df[c] = df[c].where(reg)
    df = df.drop(columns="same_mkt_dev")
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df


DERIVED8 = ["comp_effect", "n_mkt_dev", "mkt_disp"]



# %% 단조 제약 (13차 채택: 축·수). 방향 점검(eda/23)에서 '일치' 확인된 피처만
def _mono():
    chk = pd.read_csv(ROOT / "eda" / "tables" / "direction_check.csv")
    out = {}
    for d, nm in cs.rm.NAMES.items():
        ok = chk[(chk.분야 == nm) & (chk.판정 == "일치")]
        out[d] = {r["feature"]: (1 if r["기대 방향"] == "+" else -1) for _, r in ok.iterrows()}
    return out


MONO = _mono()
TRAIN_CFG["livestock"]["mono"] = True
TRAIN_CFG["fishery"]["mono"] = True


# %% 파생변수 9차 (D31·D32) — 수산물 (농·축에도 D31 적용 가능)
def add_derived9(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    f = cs.fs.frame(d)[cs.KEYS + ["date"] + [c for c in ["export_kg", "catch_ton", "catch_value_kkrw"] if c in cs.fs.frame(d).columns]].copy() if False else None
    nat = df[df.is_nat == 1][cs.KEYS + ["date"] + [c for c in ["export_kg", "catch_ton", "catch_value_kkrw"] if c in df.columns]].copy()
    nat["sgg_nm"] = "전국"
    if "export_kg" in nat:
        e3 = _rolling_mean(nat, "export_kg", "91D", min_periods=20)
        nat["_e3"] = e3
        prev = nat[cs.KEYS + ["date", "_e3"]].copy(); prev["date"] = prev.date + pd.Timedelta(days=364)
        nat = nat.merge(prev.rename(columns={"_e3": "_e3_ly"}), on=cs.KEYS + ["date"], how="left")
        nat["export_yoy"] = np.log((nat._e3 + 1) / (nat._e3_ly + 1))
    if "catch_ton" in nat and "catch_value_kkrw" in nat:
        nat["_u"] = nat.catch_value_kkrw / nat.catch_ton.where(nat.catch_ton > 0)
        prev = nat[cs.KEYS + ["date", "_u"]].copy(); prev["date"] = prev.date + pd.Timedelta(days=364)
        nat = nat.merge(prev.rename(columns={"_u": "_u_ly"}), on=cs.KEYS + ["date"], how="left")
        nat["catch_unit_yoy"] = np.log(nat._u / nat._u_ly)
    keep = [c for c in ["export_yoy", "catch_unit_yoy"] if c in nat]
    df = df.merge(nat[cs.KEYS + ["date"] + keep].replace([np.inf, -np.inf], np.nan), on=cs.KEYS + ["date"], how="left")
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df


# %% 2라운드 A: 변수 성격별 파생변수 (G1 지수류 변화율, G2 물량류 전년 동기 대비, G4 해양·양식장 평년 대비)
G1 = ["impidx_과실및채소가공품", "impidx_농산물", "impidx_item_matched", "cpi_item", "cpi_식료품", "usd_krw", "fuel_휘발유경유등유", "intl_사료곡물_옥수수대두밀"]
G2 = ["import_kg", "export_kg", "catch_ton", "catch_value_kkrw", "stock_ton", "census_heads", "auction_heads", "garak_n_trades"]
G4 = ["sea_temp_east_south_west_coast", "farm_temp_s_b", "farm_do_s_b", "farm_din_dip_s", "farm_cod_s", "farm_sal_s"]


def _lag_value(nat, col, days):
    """같은 시계열에서 days일 전(그 이전 가장 가까운 날)의 값."""
    a = nat[cs.KEYS + ["date", col]].dropna(subset=[col]).sort_values("date")
    q = nat[cs.KEYS + ["date"]].copy()
    q["_t"] = (q.date - pd.Timedelta(days=days)).astype("datetime64[ns]")
    a = a.rename(columns={"date": "_t", col: "_lag"}); a["_t"] = a._t.astype("datetime64[ns]")
    m = pd.merge_asof(q.sort_values("_t"), a, on="_t", by=cs.KEYS, direction="backward", tolerance=pd.Timedelta(days=45))
    return m.set_index(q.sort_values("_t").index)._lag.reindex(q.index).values


def typed_groups(d, nm):
    base = set(BASE[BASE.분야 == nm].feature)
    return {"G1": [c for c in G1 if c in base], "G2": [c for c in G2 if c in base], "G4": [c for c in G4 if c in base]}


def add_typed(df, d):
    scope = dict(df.attrs.get("scope", {}))
    n0 = len(df)
    nm = cs.rm.NAMES[d]
    grp = typed_groups(d, nm)
    nat = cs.fs.frame(d)
    nat = nat[cs.KEYS + ["date"] + sorted(set(sum(grp.values(), [])))].copy()
    nat["woy"], nat["year"] = nat.date.dt.isocalendar().week.astype(int), nat.date.dt.year
    new = []
    for c in grp["G1"]:
        for k in (28, 91):
            nat[f"chg{k}_{c}"] = np.log(nat[c] / _lag_value(nat, c, k)); new.append(f"chg{k}_{c}")
    for c in grp["G2"]:
        nat[f"yoy_{c}"] = np.log((nat[c] + 1) / (_lag_value(nat, c, 364) + 1)); new.append(f"yoy_{c}")
    for c in grp["G4"]:
        norm = _prior_year_mean(nat, cs.KEYS, c).rename(columns={"prior": "_n"})
        nat = nat.merge(norm, on=cs.KEYS + ["woy", "year"], how="left")
        nat[f"anom_{c}"] = nat[c] - nat._n; nat = nat.drop(columns="_n"); new.append(f"anom_{c}")
    nat[new] = nat[new].replace([np.inf, -np.inf], np.nan)
    df = df.merge(nat[cs.KEYS + ["date"] + new], on=cs.KEYS + ["date"], how="left")
    for c in new:
        src = c.split("_", 1)[1]
        if src in scope:
            scope[c] = scope[src]
    assert len(df) == n0
    df.attrs["scope"] = scope
    return df, {g: [f"chg{k}_{c}" for c in v for k in (28, 91)] if g == "G1" else [("yoy_" if g == "G2" else "anom_") + c for c in v] for g, v in grp.items()}



# %% 2라운드 채택: 수산물 G1′ (지수 수준 → 4주·13주 변화율 교체)
TYPED_ADOPTED = {"fishery": ["G1′"]}


def final_frame(d):
    """최종 구성의 데이터와 수치 피처 목록."""
    nm = cs.rm.NAMES[d]
    df = build_all(d)
    num, cats = base_sets(nm)
    num = num + ADOPTED[d]
    if TYPED_ADOPTED.get(d):
        df, G = add_typed(df, d)
        for g in TYPED_ADOPTED[d]:
            if g == "G1′":
                num = [c for c in num if c not in typed_groups(d, nm)["G1"]] + G["G1"]
            else:
                num = num + G[g]
    return df, num, cats


# %% 피처 저장·재사용 (최종 구성, 버전별)
FRAME_VER = "v1"


def cached_final_frame(d, rebuild=False):
    """final_frame 결과를 data/processed/{d}/model_frame_{ver}.parquet에 저장하고 재사용. 품목 적용 범위(scope)는 _scope__ 열로 저장."""
    import json
    path = cs.P / d / f"model_frame_{FRAME_VER}.parquet"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists() and not rebuild:
        df = pd.read_parquet(path)
        m = json.loads(meta.read_text(encoding="utf-8"))
        sc = [c for c in df.columns if c.startswith("_scope__")]
        df.attrs["scope"] = {c[len("_scope__"):]: df[c].values.astype(bool) for c in sc}
        return df.drop(columns=sc), m["num"], m["cats"]
    df, num, cats = final_frame(d)
    keep = sorted(set(cs.KEYS + ["sgg_nm", "date", "y", "is_nat", "food_group", "dev_last", "base"] + num + cats))
    out = df[[c for c in keep if c in df.columns]].copy()
    for c, ok in df.attrs.get("scope", {}).items():
        out[f"_scope__{c}"] = ok
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.attrs = {}
    out.to_parquet(path, index=False)
    meta.write_text(json.dumps({"num": num, "cats": cats, "ver": FRAME_VER}, ensure_ascii=False), encoding="utf-8")
    sc = [c for c in out.columns if c.startswith("_scope__")]
    res = out.drop(columns=sc)
    res.attrs["scope"] = {c[len("_scope__"):]: out[c].values.astype(bool) for c in sc}
    return res, num, cats
