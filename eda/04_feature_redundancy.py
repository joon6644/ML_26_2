# %% [markdown]
# # 피처 중복 점검
# 03_baseline_models.py와 같은 피처를 만들어 (1) 피처 간 상관, (2) VIF, (3) 데이터 수준 중복(같은 값 공유)을 본다.
# 실행: `python eda/04_feature_redundancy.py` → eda/04_feature_redundancy.md

# %%
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bm", ROOT / "eda" / "03_baseline_models.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)
CORR_TH = 0.7


def domain_frame(domain, common, item_map):
    daily, keys = bm.build_series(domain)
    df = bm.make_rows(daily, keys).merge(common, on="date", how="left").merge(item_map, on=["ctgry_cd", "item_cd"], how="left")
    fn, dom_cols = bm.DOMAIN_FEATURES[domain]
    df = fn(df).replace([np.inf, -np.inf], np.nan)
    df = df[df.y.notna() & (df.date >= bm.TRAIN_START) & (df.date <= bm.TRAIN_END)]
    return df, bm.NUM + dom_cols, daily, keys


def corr_pairs(df, cols):
    c = df[cols].corr(method="spearman")
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(c.loc[a, b]) >= CORR_TH]
    return pd.DataFrame(pairs, columns=["피처1", "피처2", "스피어만 상관"]).sort_values("스피어만 상관", key=abs, ascending=False)


def vif(df, cols):
    x = df[cols].sample(min(len(df), 100_000), random_state=0)
    x = x.fillna(x.median())
    x = x.loc[:, x.std() > 0]
    x = (x - x.mean()) / x.std()
    v = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return pd.Series(v, index=x.columns).sort_values(ascending=False)


def duplicate_series(daily, keys, min_overlap=200):
    """같은 도메인 안에서 가격 값이 거의 항상 같은 시계열 쌍 (타깃 중복)."""
    wide = daily.assign(key=daily[keys].astype(str).agg(" | ".join, axis=1)).pivot_table(index="date", columns="key", values="price_kg")
    out = []
    cols = list(wide.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = wide[a].notna() & wide[b].notna()
            if both.sum() >= min_overlap:
                same = (np.isclose(wide.loc[both, a], wide.loc[both, b], rtol=1e-6)).mean()
                if same >= 0.95:
                    out.append((a, b, int(both.sum()), same))
    return pd.DataFrame(out, columns=["시계열1", "시계열2", "겹친 날", "같은 값 비율"])


# %%
if __name__ == "__main__":
    common = bm.common_features()
    item_map = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str)
    garak = item_map[item_map.garak_item.fillna("") != ""].groupby("garak_item").item_nm.agg(list)
    shared = garak[garak.str.len() > 1]
    lines = ["# 피처 중복 점검", "", "> 자동 생성: `python eda/04_feature_redundancy.py` · 대상: 03 베이스라인 피처, 학습 구간",
             f"> 상관은 스피어만, |ρ| ≥ {CORR_TH} 인 쌍만 표시. VIF ≥ 10이면 선형회귀에서 계수가 불안정해질 수 있음", ""]
    names = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
    for d in ("agri", "livestock", "fishery"):
        df, cols, daily, keys = domain_frame(d, common, item_map[["ctgry_cd", "item_cd", "food_group"]])
        cp, vf, dup = corr_pairs(df, cols), vif(df, cols), duplicate_series(daily, keys)
        print(d, "상관쌍", len(cp), "| VIF>=10", (vf >= 10).sum(), "| 중복 시계열", len(dup))
        lines += [f"## {names[d]}", "", f"### 상관 |ρ| ≥ {CORR_TH}", "",
                  cp.round(3).to_markdown(index=False, disable_numparse=True) if len(cp) else "(없음)", "",
                  "### VIF (상위 10)", "", vf.head(10).round(1).rename("VIF").rename_axis("피처").reset_index().to_markdown(index=False, disable_numparse=True), "",
                  "### 같은 가격 값을 공유하는 시계열 (95% 이상 동일)", "",
                  dup.round(3).to_markdown(index=False, disable_numparse=True) if len(dup) else "(없음)", ""]
    lines += ["## 같은 가락시장 물량을 공유하는 품목 (item_map.garak_item)", "",
              shared.map(", ".join).rename("품목").reset_index().to_markdown(index=False, disable_numparse=True), ""]
    (ROOT / "eda" / "04_feature_redundancy.md").write_text("\n".join(lines), encoding="utf-8")
    print("저장: eda/04_feature_redundancy.md")
