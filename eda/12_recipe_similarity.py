# %% [markdown]
# # 레시피 기반 식재료 관계 (대체 후보 · 메뉴 추천 가능성 탐색)
# 입력: data/processed/common/recipe_basic·recipe_ingredient.parquet (MAFRA 레시피 537개), data/reference/item_map.csv
# 1) 레시피 재료명 → 가격 품목(109개) 매핑 커버리지
# 2) 함께 쓰이는 재료(1차 공동출현, PPMI) vs 비슷한 자리에 쓰이는 재료(2차 유사도 = PPMI 벡터 코사인)
# 3) 2차 유사도로 계층 군집
# 실행: `python eda/12_recipe_similarity.py` → eda/12_recipe_similarity.md

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
MIN_RECIPES = 5
ALIAS = {  # 레시피 재료명 → 가격 품목 (부분 일치보다 우선)
    "소": ["쇠고기", "소고기", "한우", "불고기", "차돌", "사태", "우둔", "양지", "등심", "안심", "갈비"],
    "돼지": ["돼지고기", "삼겹살", "목살", "앞다리", "돼지갈비", "돈육"],
    "닭": ["닭고기", "닭가슴살", "닭다리", "닭날개", "영계"],
    "계란": ["달걀", "계란", "메추리알"],
    "깐마늘(국산)": ["마늘", "다진마늘", "통마늘", "편마늘", "마늘즙"],
    "파": ["대파", "쪽파", "실파", "다진파", "파채", "송송"],
    "물오징어": ["오징어"],
    "붉은고추": ["홍고추", "붉은고추"],
    "풋고추": ["청고추", "풋고추", "청양고추", "꽈리고추"],
    "참깨": ["깨소금", "통깨", "참깨", "볶은깨"],
    "호박": ["애호박", "호박"],
    "생강": ["생강", "다진생강", "생강즙"],
    "쌀": ["쌀", "멥쌀", "밥"],
    "마른멸치": ["멸치", "잔멸치"],
    "새우": ["새우", "대하", "칵테일새우"],
    "건다시마": ["다시마"],
    "마른미역": ["미역"],
}
NOT_ITEM = {"김치", "배추김치", "묵은지", "깍두기", "김가루", "무순", "파슬리", "파프리카가루", "배즙", "콩나물", "콩가루", "두부"}


def build_matcher():
    m = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str).fillna("")
    kw = []
    for _, r in m.iterrows():
        base = re.sub(r"\(.*\)", "", r.item_nm)
        for k in {base, *[x for x in r.search_keywords.split(";") if x]}:
            if not k.endswith("값"):
                kw.append((k, r.item_nm))
    kw.sort(key=lambda x: -len(x[0]))
    alias = {a: it for it, al in ALIAS.items() for a in al}

    def match(name):
        n = re.sub(r"\s+", "", str(name))
        if n in NOT_ITEM:
            return None
        for a in sorted(alias, key=len, reverse=True):
            if a in n:
                return alias[a]
        for k, it in kw:
            if len(k) == 1:                       # 한 글자 품목(파·무·배·김·굴 …)은 정확히 같을 때만
                if n == k:
                    return it
            elif k in n:
                return it
        return None
    return match, m.set_index("item_nm").food_group.to_dict()


def main():
    rb = pd.read_parquet(P / "common" / "recipe_basic.parquet")
    ri = pd.read_parquet(P / "common" / "recipe_ingredient.parquet")
    match, fg = build_matcher()
    ri["item"] = ri.irdnt_nm.map(match)
    L = ["# 레시피 기반 식재료 관계 (자동 생성)", "", "> `python eda/12_recipe_similarity.py`", ""]

    # 1) 커버리지
    cov = ri.groupby("irdnt_ty_nm").item.apply(lambda s: s.notna().mean() * 100).round(0)
    main_ = ri[ri.irdnt_ty_nm.isin(["주재료", "부재료"])]
    full = main_.groupby("recipe_id").item.apply(lambda s: s.notna().all())
    full_main = ri[ri.irdnt_ty_nm == "주재료"].groupby("recipe_id").item.apply(lambda s: s.notna().all())
    unmatched = main_[main_.item.isna()].irdnt_nm.value_counts().head(30)
    qty = ri.irdnt_cpcty.astype(str)
    q_g = qty.str.contains(r"\d\s*(g|kg|ml|L)\b", case=False).mean() * 100
    L += ["## 1. 가격 품목 매핑 커버리지", "",
          "| 재료 구분 | 가격 품목으로 매핑된 비율 |", "|---|---|"] + [f"| {k} | {v:.0f}% |" for k, v in cov.items()] + ["",
          f"- 주재료가 모두 가격 품목인 레시피: {full_main.sum()} / {len(full_main)} ({full_main.mean() * 100:.0f}%)",
          f"- 주재료+부재료가 모두 가격 품목인 레시피: {full.sum()} / {len(full)} ({full.mean() * 100:.0f}%)",
          f"- 분량이 g·kg·ml로 적힌 재료 행: {q_g:.0f}% (나머지는 개·큰술·약간 등)",
          f"- 레시피 국가: {rb.nation_nm.value_counts().to_dict()}",
          f"- 레시피 분량: {rb.qnt.value_counts().head(5).to_dict()}", "",
          "가격 품목에 없는 주·부재료 상위 30: " + ", ".join(f"{k}({v})" for k, v in unmatched.items()), ""]
    print("\n".join(L))

    # 2) 품목 × 레시피 행렬, PPMI
    pairs = ri.dropna(subset=["item"])[["recipe_id", "item"]].drop_duplicates()
    cnt = pairs.item.value_counts()
    items = cnt[cnt >= MIN_RECIPES].index.tolist()
    X = pd.crosstab(pairs.item, pairs.recipe_id).reindex(items).values.astype(float)
    C = X @ X.T
    n_rec = X.shape[1]
    p_i = X.sum(1) / n_rec
    pmi = np.log((C / n_rec + 1e-12) / np.outer(p_i, p_i))
    ppmi = np.where(C > 0, np.maximum(pmi, 0), 0)
    np.fill_diagonal(ppmi, 0)
    V = ppmi / (np.linalg.norm(ppmi, axis=1, keepdims=True) + 1e-12)
    S2 = V @ V.T                                  # 2차 유사도: 같은 재료들과 어울리는가
    np.fill_diagonal(S2, 0)
    co = pd.DataFrame(ppmi, index=items, columns=items)
    s2 = pd.DataFrame(S2, index=items, columns=items)
    L2 = [f"## 2. 이웃 재료 (레시피 {MIN_RECIPES}개 이상 등장한 품목 {len(items)}개)", "",
          "- **함께 쓰는 재료**: 같은 레시피에 자주 같이 등장 (PPMI, 보완재)",
          "- **비슷한 자리의 재료**: 함께 쓰는 재료 구성이 비슷함 (2차 유사도, 대체재 후보)", "",
          "| 품목 (레시피 수) | 함께 쓰는 재료 | 비슷한 자리의 재료 (유사도) |", "|---|---|---|"]
    show = ["파", "양파", "깐마늘(국산)", "소", "돼지", "닭", "계란", "물오징어", "새우", "감자", "당근", "무", "배추",
            "오이", "느타리버섯", "팽이버섯", "시금치", "고등어", "부추", "깻잎", "미나리", "호박"]
    for it in [x for x in show if x in items]:
        a = co.loc[it].sort_values(ascending=False).head(4).index
        b = s2.loc[it].sort_values(ascending=False).head(4)
        L2.append(f"| {it} ({cnt[it]}) | {', '.join(a)} | {', '.join(f'{k} {v:.2f}' for k, v in b.items())} |")
    L2.append("")

    # 3) 계층 군집 (2차 유사도)
    D = 1 - (S2 + np.eye(len(items)))
    D = np.clip((D + D.T) / 2, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    lab = fcluster(Z, t=10, criterion="maxclust")
    cl = pd.DataFrame({"item": items, "cl": lab, "fg": [fg.get(i, "") for i in items]})
    L2 += ["## 3. 2차 유사도 계층 군집 (10개)", "", "| 군집 | 품목 (식재료 계열) |", "|---|---|"]
    for k, g in cl.groupby("cl"):
        L2.append(f"| {k} | " + ", ".join(f"{r.item}({r.fg})" for r in g.itertuples()) + " |")
    print("\n".join(L2))
    (ROOT / "eda" / "12_recipe_similarity.md").write_text("\n".join(L + L2), encoding="utf-8")


if __name__ == "__main__":
    main()
