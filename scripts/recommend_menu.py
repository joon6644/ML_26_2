"""이번 주 메뉴 추천 (개인 소비자 = 소매가 기준)

    python scripts/recommend_menu.py            # 재료 상태 계산(모델 학습·예측) → 추천 출력
    python scripts/recommend_menu.py --cached   # 저장된 재료 상태로 추천만

1) 재료 상태 (data/processed/menu/ingredient_state.parquet): 신선식품 품목별
   - 가격 수준: 최근 7일 평균이 최근 1년 중 몇 백분위인가 (낮을수록 싼 편)
   - 4주 전망: 향후 28일 평균 / 최근 7일 평균 − 1 예측 (분야별 LightGBM, 전국 소매 시계열)
2) 재료 점수 = 가격 수준 점수 + 전망 점수 (각 −1 / 0 / +1)
   - 가격 수준: 하위 30% 이하 +1 (싼 편), 상위 30% 이상 −1 (비싼 편)
   - 전망: −2% 이하 +1 (하락 예상), +2% 이상 −1 (상승 예상), 그 사이 0 (보합). 보합 기준은 모델 평가와 같은 ±2%
3) 레시피 점수 = 가격이 연결된 신선식품 재료 점수의 평균 (양념·생필품은 점수에 넣지 않음)
4) 메뉴 점수 = 그 메뉴 레시피 점수의 최고값. 같은 메뉴는 최고점 레시피들 중 순수 랜덤으로 1개
5) 메뉴 순위: 점수 높은 순 (조회수 무관). 동점이면 점수에 들어간 재료 수가 많은 순, 그다음 랜덤
"""
import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import build_features as bf
from common import PROCESSED as P

OUT = P / "menu"
TOP, MORE = 3, 30


def item_state():
    rows = []
    for d in ("agri", "livestock", "fishery"):
        ds = bf.add_history_and_target(pd.read_parquet(P / d / "dataset.parquet"))
        ds["woy"] = ds.date.dt.isocalendar().week.astype("int16")
        num = [c for c in bf.COMMON + bf.DOMAIN[d] + bf.HIST + ["woy"] if c in ds]
        tr = ds.y.notna()
        X = ds[num].astype("float32").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X[tr].mean()).fillna(0)
        Xc = pd.DataFrame(MinMaxScaler().fit(X[tr]).transform(X), columns=num, index=ds.index).astype("float32")
        for c in bf.CAT:
            Xc[c] = ds[c].astype(str).astype(pd.CategoricalDtype(sorted(ds[c].astype(str).unique())))
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(Xc[tr], ds.y[tr], categorical_feature=bf.CAT)
        last = ds.date.max()
        now = ds[(ds.se_nm == "소매") & (ds.date > last - pd.Timedelta(days=7))]
        now = now.sort_values("date").groupby(bf.KEYS, observed=True).tail(1)
        now = now.assign(pred=m.predict(Xc.loc[now.index]))
        # 가격 수준: 시계열별 최근 1년 7일 평균 분포에서 현재 7일 평균의 백분위
        hist = ds[(ds.se_nm == "소매") & (ds.date > last - pd.Timedelta(days=365))]
        pct = hist.groupby(bf.KEYS, observed=True).base.apply(lambda s: (s.dropna() <= s.dropna().iloc[-1]).mean() if s.notna().sum() >= 30 else np.nan)
        now = now.merge(pct.rename("pct").reset_index(), on=bf.KEYS, how="left")
        g = now.groupby("item_nm").agg(pred=("pred", "median"), pct=("pct", "median"), n_series=("pred", "size"), base=("base", "median"))
        rows.append(g.assign(domain=d, as_of=last))
        print(f"  {d}: 기준일 {last:%Y-%m-%d}, 품목 {len(g)}개", flush=True)
    st = pd.concat(rows).reset_index()
    st["level"] = np.select([st.pct <= 0.3, st.pct >= 0.7], [1, -1], 0)
    st["trend"] = np.select([st.pred <= -0.02, st.pred >= 0.02], [1, -1], 0)
    st.loc[st.pct.isna(), "level"] = 0
    st["score"] = st.level + st.trend
    st.to_parquet(OUT / "ingredient_state.parquet", index=False)
    return st


def reason(r):
    lv = {1: "1년 중 싼 편", -1: "1년 중 비싼 편", 0: "평소 수준"}[r.level]
    tr = {1: f"4주 하락 예상({r.pred:+.0%})", -1: f"4주 상승 예상({r.pred:+.0%})", 0: "4주 보합"}[r.trend]
    return f"{r.ingredient} {lv}·{tr}"


def recommend(st, seed=None):
    rng = np.random.default_rng(seed)
    rec = pd.read_parquet(OUT / "recipes.parquet")
    rec = rec[rec.is_menu]
    ing = pd.read_parquet(OUT / "recipe_ingredients.parquet")
    ing = ing[(ing["class"] == "식재료") & (ing.price_source == "신선식품") & ing.recipe_id.isin(rec.recipe_id)]
    ing = ing.merge(st[["item_nm", "score", "level", "trend", "pred", "pct"]], left_on="price_item", right_on="item_nm")
    ing = ing.drop_duplicates(["recipe_id", "price_item"])
    rs = ing.groupby("recipe_id").agg(score=("score", "mean"), n_scored=("score", "size")).reset_index()
    rs = rs.merge(rec[["recipe_id", "menu_name", "name", "url", "time", "servings"]], on="recipe_id")
    rs["rand"] = rng.random(len(rs))
    best = rs[rs.score == rs.groupby("menu_name").score.transform("max")]
    pick = best.sort_values("rand").groupby("menu_name").head(1)                     # 같은 메뉴: 최고점 레시피 중 순수 랜덤
    pick = pick.assign(rand2=rng.random(len(pick))).sort_values(["score", "n_scored", "rand2"], ascending=[False, False, True])
    pick = pick.head(MORE).reset_index(drop=True)
    why = ing.sort_values("score", ascending=False).groupby("recipe_id").apply(
        lambda g: " / ".join(reason(r) for r in g.head(3).itertuples()), include_groups=False)
    pick["reason"] = pick.recipe_id.map(why)
    pick.insert(0, "rank", range(1, len(pick) + 1))
    return pick[["rank", "menu_name", "score", "n_scored", "reason", "time", "servings", "url", "name"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    st = pd.read_parquet(OUT / "ingredient_state.parquet") if a.cached else item_state()
    top = recommend(st, a.seed)
    top.to_csv(OUT / "recommend_latest.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"\n== 이번 주 추천 {TOP}개 (기준일 {st.as_of.max():%Y-%m-%d}) ==")
    print(top.head(TOP)[["rank", "menu_name", "score", "reason"]].to_string(index=False))
    print(f"\n== 더 보기 {MORE}개 ==")
    print(top[["rank", "menu_name", "score", "n_scored", "reason"]].to_string(index=False))
    sys.exit(0)
