# %% [markdown]
# # 최종 피처 선택 평가 (data/reference/feature_selection.csv)
# 비교: ① 전체 피처 (features.parquet 전부 + 신규 3개) ② CSV에 남은 피처
# 신규 피처: days_to_seollal, days_to_chuseok (dataset.parquet), grp_r28 = 같은 날 같은 식재료 계열 다른 시계열들의 r28 평균 (자기 제외)
# 모델: LightGBM, 전국, 학습 2016~2022 / 테스트 2023~2024, 시드 3개 평균. 베이스라인 1 (오늘 가격 = 향후 4주 평균) 함께 보고
# 실행: `python eda/21_final_selection.py` → eda/21_final_selection.md

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
SPLIT = ("2016-01-01", "2022-12-01", "2023-01-01", "2024-12-31")
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
SEEDS = (0, 1, 2)
KEYS = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
# 같은 역할 세부 피처 통합: 이름 → (분야, 묶는 피처, 방식). raw = 같은 단위 평균, norm = 학습기간 평균 1로 맞춘 뒤 평균. 결측은 있는 값만 평균
COMPOSITES = {
    "fuel_휘발유경유등유": (["agri", "livestock", "fishery"], ["fuel_보통휘발유", "fuel_자동차용_경유", "fuel_실내등유"], "norm"),
    "wx_temp_avg_nat_prod": (["agri"], ["wx_temp_avg", "prodarea_temp_avg"], "raw"),
    "wx_rain_nat_prod": (["agri"], ["wx_rain", "prodarea_rain"], "raw"),
    "wx_sun_solar_hr_nat_prod": (["agri"], ["wx_solar_rad", "wx_sunshine_hr", "prodarea_sunshine_hr"], "norm"),
    "impidx_도축육_가금육": (["livestock"], ["impidx_도축육", "impidx_가금육"], "raw"),
    "intl_사료곡물_옥수수대두밀": (["livestock"], ["intl_옥수수", "intl_대두", "intl_소맥"], "norm"),
    "sea_temp_east_south_west_coast": (["fishery"], ["sea_temp_east", "sea_temp_south", "sea_temp_west", "coast_temp_avg"], "raw"),
    "farm_din_dip_s": (["fishery"], ["farm_din_s", "farm_dip_s"], "norm"),
    "farm_temp_s_b": (["fishery"], ["farm_temp_s", "farm_temp_b"], "raw"),
    "farm_do_s_b": (["fishery"], ["farm_do_s", "farm_do_b"], "raw"),
    "impidx_수산가공품_수산물가공품": (["fishery"], ["impidx_수산가공품", "impidx_수산물가공품"], "raw"),
    "impidx_냉동수산물_냉동건조수산물": (["fishery"], ["impidx_냉동수산물", "impidx_냉동건조수산물"], "raw"),
}


def frame(d):
    f = pd.read_parquet(P / d / "features.parquet")
    hol = pd.read_parquet(P / d / "dataset.parquet", columns=KEYS + ["date", "days_to_seollal", "days_to_chuseok"])
    f = f.merge(hol, on=KEYS + ["date"], how="left")
    g = f.groupby(["date", "food_group"]).r28
    s, n = g.transform("sum"), g.transform("count")
    f["grp_r28"] = (s - f.r28.fillna(0)) / (n - f.r28.notna().astype(int))   # 자기 시계열 제외
    f.loc[(n - f.r28.notna().astype(int)) <= 0, "grp_r28"] = np.nan
    tr = (f.date >= SPLIT[0]) & (f.date <= SPLIT[1])
    for new, (ds, cols, how) in COMPOSITES.items():
        if d in ds:
            X = f[cols] if how == "raw" else f[cols] / f.loc[tr, cols].mean()
            f[new] = X.mean(axis=1)
    return add_item_matched(add_events(f, d), d)


# 사건 감지 피처: 하루 평균으로는 묽어지는 태풍·강풍·감염병을 "최근 N일 안에 있었는지·얼마나"로 표현
DISEASE_BY_ITEM = {"계란": ["고병원성조류인플루엔자"], "닭": ["고병원성조류인플루엔자"],
                   "돼지": ["아프리카돼지열병", "돼지열병"], "수입 돼지고기": ["아프리카돼지열병", "돼지열병"],
                   "소": ["구제역", "럼피스킨병"], "수입 소고기": ["구제역", "럼피스킨병"], "우유": ["구제역", "럼피스킨병"]}
_EVENTS = {}


def gust20_share_14d():
    """전국 ASOS 관측소 중 그날 순간최대풍속 20m/s 이상인 비율 → 최근 14일 최댓값 (태풍·강풍 감지)."""
    st = pd.read_parquet(P / "common" / "weather_station_daily.parquet", columns=["date", "wind_gust"])
    s = st.assign(s=st.wind_gust >= 20).groupby("date").s.mean().asfreq("D").rolling(14, min_periods=1).max()
    return s.rename("wx_gust20_station_share_14d").reset_index()


def disease_item_cases_28d():
    """품목에 맞는 주요 가축전염병의 최근 28일 발생 건수 (AI는 2020년 이후 두수가 비어 있어 건수 사용)."""
    e = pd.read_parquet(P / "livestock" / "disease_events.parquet", columns=["date", "disease_std"])
    idx = pd.date_range("2014-01-01", e.date.max() + pd.Timedelta(days=365))
    out = []
    for item, ds in DISEASE_BY_ITEM.items():
        c = e[e.disease_std.isin(ds)].groupby("date").size().reindex(idx, fill_value=0).rolling(28, min_periods=1).sum()
        out.append(pd.DataFrame({"date": idx, "item_nm": item, "disease_item_cases_28d": c.values}))
    return pd.concat(out, ignore_index=True)


def buoy_storm_share_7d():
    """해양기상부이(2016년부터 연속 운영 17곳) 중 그날 풍속 14m/s 이상이 3시간 이상인 부이 비율(풍랑주의보 수준) → 최근 7일 평균.
    출항 통제·조업 중단으로 어획 공급이 줄어드는 날이 최근 일주일에 얼마나 있었는지."""
    import glob
    h = pd.concat([pd.read_csv(f, encoding="cp949", usecols=["지점", "일시", "풍속(m/s)"]) for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "kma_buoy" / "*.csv")))])
    h = h.rename(columns={"지점": "stn_id", "일시": "time", "풍속(m/s)": "wind"}).drop_duplicates(["stn_id", "time"])
    h["date"] = (pd.to_datetime(h.time) - pd.Timedelta(minutes=1)).dt.normalize()
    core = pd.read_csv(ROOT / "data" / "reference" / "kma_buoy_station.csv").stn_id
    h = h[h.stn_id.isin(core)]
    d = h.groupby(["stn_id", "date"]).agg(n=("wind", "count"), strong=("wind", lambda w: (w >= 14).sum())).reset_index()
    d = d[d.n >= 12]
    s = (d.strong >= 3).groupby(d.date).mean().asfreq("D").rolling(7, min_periods=3).mean()
    return s.rename("buoy_storm_share_7d").reset_index()


# 품목 대응: 가설이 그 품목에만 성립하는 피처는 대응 품목에만 값을 두고, 나머지 품목은 "해당 없음"(-1, 통합 모델 fit에서 처리)
AQUA = ["가리비", "굴", "꼬막", "바지락", "전복", "홍합", "김", "건다시마", "마른미역"]                   # 양식(또는 갯벌 양식) 품목
FISH_PROC = ["마른멸치", "마른오징어", "북어", "멸치액젓", "새우젓", "천일염", "건다시마", "마른미역", "고등어필렛"]   # 가공 수산물
AGRI_IMPORT = {"곡류": ["쌀", "찹쌀", "현미"], "맥류및잡곡": ["보리쌀", "메밀", "혼식곡"], "콩류": ["녹두", "콩", "팥"]}   # 과일 계열은 아래에서 food_group으로
AGRI_NOT_FRUITVEG = ["곡류", "두류", "견과·종실류"]                                                   # 과실·채소 가공품 수입가와 무관한 계열
SCOPE = {   # 피처 → (분야, 대상 품목 판정 함수)
    "pig_auction_price_kg": ("livestock", lambda f: f.item_nm.isin(["돼지", "수입 돼지고기"])),
    "intl_사료곡물_옥수수대두밀": ("livestock", lambda f: f.item_nm != "우유"),   # 원유가격 연동제로 사료값과 무관
    "impidx_과실및채소가공품": ("agri", lambda f: ~f.food_group.isin(AGRI_NOT_FRUITVEG)),
    **{c: ("fishery", lambda f: f.item_nm.isin(AQUA)) for c in ["farm_cod_s", "farm_do_s_b", "farm_temp_s_b", "farm_ss_s", "farm_din_dip_s", "farm_sal_s"]},
}


def add_item_matched(f, d):
    """품목에 대응하는 수입물가 하나로 (농산물: 곡류·맥류잡곡·콩류·과일, 수산물: 신선 / 가공)."""
    if d == "agri":
        m = pd.Series(np.nan, index=f.index)
        for grp, items in AGRI_IMPORT.items():
            m[f.item_nm.isin(items)] = f.loc[f.item_nm.isin(items), f"impidx_{grp}"]
        fruit = f.food_group.str.startswith("과일")
        m[fruit] = f.loc[fruit, "impidx_과일"]
        f["impidx_item_matched"] = m
    elif d == "fishery":
        proc = f.item_nm.isin(FISH_PROC)
        f["impidx_item_matched"] = np.where(proc, f["impidx_수산가공품_수산물가공품"], f["impidx_신선수산물"])
    return f


def in_scope(f, d):
    """{피처: 대상 품목 여부 bool 배열} — 통합 모델 fit에서 대상이 아닌 행을 -1로 둠."""
    out = {c: fn(f).values for c, (dd, fn) in SCOPE.items() if dd == d}
    if d == "agri" and "impidx_item_matched" in f:
        out["impidx_item_matched"] = f.item_nm.isin(sum(AGRI_IMPORT.values(), [])).values | f.food_group.str.startswith("과일").values
    return out


def add_events(f, d):
    """분야별 사건 감지 피처를 붙임 (지역 행·전국 행 모두 같은 날짜·품목 기준)."""
    if d not in _EVENTS:
        _EVENTS[d] = {"agri": gust20_share_14d, "livestock": disease_item_cases_28d, "fishery": buoy_storm_share_7d}[d]()
    ev = _EVENTS[d]
    f = f.merge(ev, on=[c for c in ev.columns if c in ("date", "item_nm")], how="left")
    if d == "livestock":
        f["disease_item_cases_28d"] = f.disease_item_cases_28d.fillna(0)
    return f


def fit(f, num, cats, tr, te, seed):
    X = f[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr].mean()).fillna(0)
    Xc = pd.DataFrame(MinMaxScaler().fit(X[tr]).transform(X), columns=num, index=f.index).astype("float32")
    for c in cats:
        Xc[c] = f[c].astype(str).astype(pd.CategoricalDtype(sorted(f[c].astype(str).unique())))
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(Xc[tr], f.y[tr], categorical_feature=cats)
    p, y = m.predict(Xc[te]), f.y[te].values
    sv = None
    if seed == 0:
        idx = np.random.default_rng(0).choice(np.where(te)[0], min(20000, te.sum()), replace=False)
        v = np.abs(m.predict(Xc.iloc[idx], pred_contrib=True)[:, :-1]).mean(0)
        sv = pd.Series(v / v.sum() * 100, index=Xc.columns)
    return float(np.sqrt(np.mean((p - y) ** 2))), float(np.mean(np.abs(p - y))), float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), sv


def runs(f, num, cats, tr, te):
    r = [fit(f, num, cats, tr, te, s) for s in SEEDS]
    a = np.array([x[:3] for x in r])
    return a.mean(0), a.std(0), r[0][3]


def main():
    t_all = time.time()
    sel = pd.read_csv(ROOT / "data" / "reference" / "feature_selection.csv")
    L = ["# 최종 피처 선택 평가", "", "> 자동 생성: `python eda/21_final_selection.py` · LightGBM 시드 3개 평균, 학습 2016~2022 / 테스트 2023~2024",
         "> 전체 = features.parquet 전체 + 신규 3개(설·추석까지 일수, 같은 계열 4주 변화율). 최종 = feature_selection.csv에 남은 피처", ""]
    rows = []
    for d, nm in NAMES.items():
        t = time.time()
        f = frame(d)
        t0, t1, s0, s1 = SPLIT
        tr = ((f.date >= t0) & (f.date <= t1)).values
        te = ((f.date >= s0) & (f.date <= s1)).values
        allnum = [c for c in f.columns if c not in bf.ID + ["base", "y"]]
        keep = sel[(sel.분야 == nm) & (sel.그룹 != "통합 모델 전용")].feature.tolist()   # 전국 전용 모델: 통합 모델 전용 피처 제외
        cats = [c for c in bf.CAT if c in keep]
        num = [c for c in keep if c not in bf.CAT]
        miss = [c for c in num if c not in f.columns]
        assert not miss, miss
        y, dev = f.y.values[te], np.expm1(f.dev_last.fillna(0).values[te])
        base = (np.sqrt(np.mean((dev - y) ** 2)), np.mean(np.abs(dev - y)), 1 - np.sum((y - dev) ** 2) / np.sum((y - y.mean()) ** 2))
        full = runs(f, allnum, bf.CAT, tr, te)
        fin = runs(f, num, cats, tr, te)
        diff, noise = fin[0][0] - full[0][0], 2 * np.hypot(full[1][0], fin[1][0])
        verdict = "유지 (시드 잡음 이내)" if abs(diff) <= noise else ("나빠짐" if diff > 0 else "좋아짐")
        for lab, n, r in [("베이스라인 1 (오늘 가격 유지)", "-", (base, (0, 0, 0))), (f"전체 피처", len(allnum) + 3, full[:2]),
                          (f"최종 선택", len(num) + len(cats), fin[:2])]:
            rows.append({"분야": nm, "모델": lab, "피처 수": n, "RMSE": round(r[0][0], 5), "± (시드)": round(r[1][0], 5),
                         "MAE": round(r[0][1], 5), "R²": round(r[0][2], 4)})
        rows[-1]["ΔRMSE vs 전체"], rows[-1]["판정"] = round(diff, 5), verdict
        new = fin[2].reindex(["days_to_seollal", "days_to_chuseok", "grp_r28"]).round(2)
        rank = fin[2].rank(ascending=False).reindex(new.index).astype(int)
        top = fin[2].sort_values(ascending=False).head(12)
        L += [f"## {nm}", "", f"- 신규 피처 SHAP: " + ", ".join(f"{k} {v:.2f}% ({rank[k]}위/{len(fin[2])})" for k, v in new.items()),
              f"- 최종 모델 SHAP 상위 12: " + ", ".join(f"{k} {v:.1f}%" for k, v in top.items()), ""]
        print(f"{nm}: 전체 {full[0][0]:.5f} → 최종 {fin[0][0]:.5f} ({diff:+.5f}, {verdict}), {time.time() - t:.0f}초", flush=True)
    R = pd.DataFrame(rows)
    L = L[:4] + ["## 결과", "", R.fillna("").to_markdown(index=False), ""] + L[4:] + [f"> 전체 소요 {time.time() - t_all:.0f}초"]
    (ROOT / "eda" / "21_final_selection.md").write_text("\n".join(L), encoding="utf-8")
    R.to_csv(ROOT / "eda" / "tables" / "final_selection.csv", encoding="utf-8-sig", index=False)


if __name__ == "__main__":
    main()
