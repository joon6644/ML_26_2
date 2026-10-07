"""메뉴 추천용 레시피 정리 (만개의레시피) → data/processed/menu/

    python scripts/build_menu.py      # collect_manrecipe.py 다음에 실행

- 재료는 "들어가는가"만 쓴다. 분량을 g·가격으로 바꾸지 않는다 (만드는 법은 원문 링크로).
- 재료명 정규화 → 양념(가격 판단에서 제외) / 식재료로 구분 → 가격 데이터 연결
    신선식품: KAMIS 109개 품목(data/reference/item_map.csv), 생필품: 한국소비자원 생필품 가격(data/raw/price_go)
- 메뉴명 정리: 말머리·수식어·요리사 이름을 빼고 대표 메뉴명(menu_name)을 만든다
출력: recipes.parquet (레시피 1행), recipe_ingredients.parquet (레시피 × 재료), ingredient_map.csv (재료명 → 분류·가격 품목)
"""
import json
import re

import pandas as pd

from common import PROCESSED, RAW, ROOT

OUT = PROCESSED / "menu"
REF = ROOT / "data" / "reference"

# 양념·기본 조미료: 소량·상비품이라 이번 주 장보기 판단에서 제외
SEASONING = set("""
소금 꽃소금 맛소금 굵은소금 천일염 설탕 흑설탕 황설탕 물 간장 진간장 국간장 양조간장 조림간장 맛간장 저염간장 참기름 들기름 식용유 포도씨유 카놀라유 올리브유 올리브오일
후추 후춧가루 통후추 깨 통깨 깨소금 참깨 볶은깨 고추장 된장 쌈장 춘장 올리고당 물엿 조청 꿀 굴소스 맛술 미림 청주 소주 식초 사과식초 매실액 매실청
마요네즈 케첩 케찹 토마토케첩 머스타드 머스터드 다시다 소고기다시다 치킨스톡 미원 연두 액젓 멸치액젓 까나리액젓 피시소스 새우젓 고춧가루 고추가루
전분 감자전분 녹말 녹말가루 전분가루 부침가루 튀김가루 빵가루 카레가루 고추기름 레몬즙 파슬리 파슬리가루 생강가루 마늘가루 계피가루 버터 육수 멸치육수 다시마육수
얼음 얼음물 찬물 기름 식용류 해바라기유 매실액기스 검은깨 월계수잎 페페론치노 두반장 쌀뜨물 다시물 깨가루 후추가루 맛소금 설탕물 뜨거운물 우스터소스 돈까스소스 스테비아 알룰로스 데리야끼소스 스리라차 핫소스 쯔유 혼다시 가쓰오부시 와사비 연겨자 겨자 머스타드소스 바질
""".split())

# 재료명 → 가격 품목 (신선식품 KAMIS 품목명). 긴 별칭부터 부분 일치
FRESH_ALIAS = {
    "계란": ["계란", "달걀", "메추리알"], "우유": ["우유"],
    "소": ["소고기", "쇠고기", "한우", "차돌", "우삼겹", "불고기용", "국거리", "사태", "양지", "등심", "안심", "채끝", "부채살", "다짐육", "다진소고기", "소불고기"],
    "돼지": ["돼지고기", "삼겹살", "목살", "앞다리", "뒷다리", "대패", "돈육", "다진돼지고기", "돼지갈비", "항정"],
    "닭": ["닭", "닭고기", "닭가슴살", "닭다리", "닭안심", "닭봉", "닭날개"],
    "파": ["다진파", "대파", "쪽파", "실파", "파"], "깐마늘(국산)": ["다진마늘", "마늘", "편마늘", "통마늘", "마늘쫑"],
    "풋고추": ["청양고추", "청량고추", "땡초", "풋고추", "꽈리고추", "청고추", "오이고추", "고추"], "붉은고추": ["홍고추", "붉은고추"],
    "호박": ["애호박", "단호박", "호박"], "생강": ["생강", "다진생강"], "물오징어": ["오징어"],
    "마른오징어": ["진미채", "마른오징어"], "마른멸치": ["멸치", "잔멸치", "볶음멸치", "국물멸치"],
    "건다시마": ["다시마"], "마른미역": ["미역"], "북어": ["북어", "황태"], "명태": ["동태", "명태", "코다리"],
    "새우": ["새우", "칵테일새우", "냉동새우"], "굴": ["굴", "생굴"], "김": ["김가루", "조미김", "김밥김", "김"], "쌀": ["쌀", "밥", "햅쌀", "찬밥", "공기밥", "즉석밥", "햇반"],
    "느타리버섯": ["느타리"], "팽이버섯": ["팽이"], "새송이버섯": ["새송이"], "배추": ["알배추", "알배기배추", "배추"],
    "무": ["무"], "콩": ["콩", "서리태", "검은콩"], "토마토": ["토마토"], "방울토마토": ["방울토마토"],
}
NOT_FRESH = {"배추김치", "김치", "묵은지", "깍두기", "콩나물", "숙주", "두부", "순두부", "연두부", "콩가루", "무말랭이", "단무지", "파프리카가루",
             "파슬리", "파마산", "파스타", "파래", "김치국물", "깻잎김치", "호박잎", "쌀국수", "쌀떡", "떡", "떡국떡", "떡볶이떡", "밥솥", "흰밥", "무순",
             "새우젓", "멸치액젓", "어묵", "맛살", "게맛살", "다시마육수", "멸치육수", "미역줄기"}
# 생필품(가공식품): 재료명 키워드 → 생필품 상품명 키워드
PROCESSED_FOOD = {
    "두부": (["두부", "순두부", "연두부"], ["두부"]), "콩나물": (["콩나물", "숙주"], ["콩나물"]), "김치": (["김치", "묵은지", "배추김치"], ["포기김치", "배추김치"]),
    "어묵": (["어묵"], ["어묵"]), "맛살": (["맛살", "게맛살", "크래미"], ["맛살"]), "햄": (["햄", "스팸", "베이컨", "소시지", "소세지", "비엔나"], ["햄", "스팸", "베이컨", "소시지"]),
    "참치": (["참치"], ["참치"]), "만두": (["만두"], ["만두"]), "라면": (["라면", "라면사리"], ["라면"]), "치즈": (["치즈"], ["치즈"]),
    "밀가루": (["밀가루", "중력분", "박력분"], ["밀가루"]), "국수": (["소면", "국수", "중면"], ["소면", "국수"]), "당면": (["당면"], ["당면"]),
    "떡": (["떡볶이떡", "떡국떡", "떡", "쌀떡"], ["떡볶이"]), "파스타": (["파스타", "스파게티"], ["스파게티", "파스타"]),
}


def clean_ing(name):
    n = re.sub(r"\(.*?\)|\[.*?\]|<.*?>", "", str(name))
    n = re.sub(r"(약간|적당량|조금|기호에\s*따라|선택|옵션|생략가능|또는.*)$", "", n.strip())
    n = re.sub(r"^(채썬|썬|삶은|데친|냉동|신선한|손질된|불린|말린|국산|수입산?)\s*", "", n)
    return re.sub(r"\s+", "", n)


def build_maps(names, products):
    fresh = sorted(((a, it) for it, al in FRESH_ALIAS.items() for a in al), key=lambda x: -len(x[0]))
    items = pd.read_csv(REF / "item_map.csv", dtype=str).fillna("")
    base = sorted(((re.sub(r"\(.*\)", "", r.item_nm), r.item_nm) for r in items.itertuples() if len(re.sub(r"\(.*\)", "", r.item_nm)) >= 2),
                  key=lambda x: -len(x[0]))
    rows = []
    for n in names:
        cls, item, src = "식재료", None, None
        if n in SEASONING or (n.endswith(("소스", "액젓", "가루", "육수", "오일", "간장", "식초", "시럽")) and n not in ("김가루", "콩가루", "들깨가루")):
            cls = "양념"
        elif n not in NOT_FRESH and not n.endswith("김치"):
            for a, it in fresh:
                if (n == a) if len(a) == 1 else (a in n):
                    item, src = it, "신선식품"
                    break
            if item is None:
                for b, it in base:
                    if b in n:
                        item, src = it, "신선식품"
                        break
        if cls == "식재료" and item is None:
            for g, (kws, pkws) in PROCESSED_FOOD.items():
                if any(k in n for k in kws) and any(any(p in pn for pn in products) for p in pkws):
                    item, src = g, "생필품"
                    break
        rows.append({"ingredient": n, "class": cls, "price_item": item, "price_source": src})
    return pd.DataFrame(rows)


DISH = ("볶음밥", "비빔밥", "덮밥", "주먹밥", "김밥", "볶음", "조림", "장조림", "무침", "찌개", "전골", "국", "탕", "전", "부침", "구이", "찜", "밥",
        "국수", "라면", "우동", "면", "파스타", "스파게티", "리조또", "샐러드", "떡볶이", "볶이", "죽", "튀김", "스테이크", "토스트", "샌드위치", "피자", "카레",
        "짜글이", "나물", "쌈", "롤", "수프", "스프", "오믈렛", "말이", "동그랑땡", "꼬치", "냉채", "겉절이", "장아찌", "피클", "소박이", "김치", "빵", "케이크",
        "쿠키", "강정", "불고기", "육회", "오이지", "깐풍기", "소바", "브라우니", "푸딩", "쌈장", "어니언링", "깍두기", "절임", "잡채", "동파육", "타르트", "버거", "카나페", "부르스게타", "브루스케타", "크럼블", "오일파스타", "올리오", "볶음우동", "까스", "가스", "치킨", "볶음면", "비빔면", "냉면", "짬뽕", "짜장", "탕수육", "만두", "그라탕", "또띠아", "부리또", "타코", "핫도그", "와플")
NOT_DISH = {"버전", "완전", "도전", "사전", "충전", "설탕", "No설탕", "흑설탕", "전전", "원전", "발전", "대전", "고전", "야채", "채소"}
NOISE = re.compile(r"초간단|간단|자취|혼밥|백종원|백주부|황금|레시피|만들기|만드는\s*법|끓이는\s*법|맛있게|맛있는|요리|반찬|메뉴|추천|꿀팁|비법|1인분|한그릇|한끼|초스피드|야매|노오븐|신혼밥상|영상")


def menu_name(t):
    t = str(t)
    quoted = re.findall(r"[<\[【'\"‘“]\s*([^<>\[\]【】'\"‘’“”]{2,20}?)\s*[>\]】'\"’”]", t)
    body = re.sub(r"\[.*?\]|\(.*?\)|【.*?】|<.*?>|#\S+", " ", t)
    def pick(text):
        toks = [w for w in re.split(r"[^\w가-힣]+", text) if w and not re.fullmatch(r"\d+", w)]
        for k, w in enumerate(toks):
            core = NOISE.sub("", w)
            if len(core) >= 2 and core.endswith(DISH) and core not in NOT_DISH:
                if core in DISH and k > 0 and len(NOISE.sub("", toks[k - 1])) >= 2:   # '팽이버섯 덮밥' → 팽이버섯덮밥
                    return NOISE.sub("", toks[k - 1]) + core
                nxt = NOISE.sub("", toks[k + 1]) if k + 1 < len(toks) else ""
                if nxt in DISH:                                                          # '콩나물 무침' → 콩나물무침
                    return core + nxt
                return core
        return None
    for q in quoted:
        m = pick(q)
        if m:
            return m
    return pick(body) or pick(t)


UNIFY = [("달걀", "계란"), ("오뎅", "어묵"), ("소세지", "소시지"), ("쏘야", "소시지야채볶음"), ("챱", "찹"), ("돈까스", "돈가스"),
         ("마늘종", "마늘쫑"), ("뭇국", "무국"), ("만둣국", "만두국"), ("닭도리탕", "닭볶음탕"), ("고추가루", "고춧가루"), ("스파게티", "파스타")]


def unify(m):
    """같은 메뉴가 표기 차이로 갈라지지 않게 통일 (달걀→계란, 오뎅→어묵, 스파게티→파스타 등)."""
    for a, b in UNIFY:
        m = m.replace(a, b)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(RAW / "manrecipe" / "recipes.parquet")
    ids = pd.read_csv(RAW / "manrecipe" / "ids.csv", dtype={"recipe_id": str})
    r = r.merge(ids[["recipe_id", "keywords"]], on="recipe_id", how="left")
    products = pd.read_csv(next((RAW / "price_go").glob("*.csv")), encoding="cp949").상품명.unique().tolist()

    ing = []
    for x in r.itertuples():
        pairs = json.loads(x.ingredients)
        if not pairs and isinstance(x.ingredients_text, str):
            pairs = [(p, "") for p in re.split(r"[,♣■:]", x.ingredients_text)]
        for raw, _ in pairs:
            n = clean_ing(raw)
            if n:
                ing.append((x.recipe_id, raw.strip(), n))
    ing = pd.DataFrame(ing, columns=["recipe_id", "raw", "ingredient"]).drop_duplicates(["recipe_id", "ingredient"])
    imap = build_maps(sorted(ing.ingredient.unique()), products)
    imap = imap.merge(ing.ingredient.value_counts().rename("n_recipes"), left_on="ingredient", right_index=True).sort_values("n_recipes", ascending=False)
    imap.to_csv(OUT / "ingredient_map.csv", index=False, encoding="utf-8-sig")
    ing = ing.merge(imap[["ingredient", "class", "price_item", "price_source"]], on="ingredient")
    ing.to_parquet(OUT / "recipe_ingredients.parquet", index=False)

    food = ing[ing["class"] == "식재료"]
    cov = food.groupby("recipe_id").agg(n_food=("ingredient", "size"), n_priced=("price_item", lambda s: s.notna().sum()),
                                        food_items=("ingredient", lambda s: ";".join(s)),
                                        price_items=("price_item", lambda s: ";".join(dict.fromkeys(s.dropna()))))
    r["menu_name"] = r.name.map(menu_name)                                   # 규칙 기반 자동 추출
    ov = pd.read_csv(REF / "menu_name_override.csv", dtype=str)             # 전수 검토 수정표 (자동 추출이 틀리거나 비어 있던 것)
    r["menu_name"] = r.recipe_id.map(ov.set_index("recipe_id").menu_name).fillna(r.menu_name)
    r["is_menu"] = r.menu_name.notna() & (r.menu_name != "제외")           # 제외 = 보관법·소스·제목 없음 등 메뉴가 아닌 글
    r.loc[~r.is_menu, "menu_name"] = None
    r["menu_name"] = r.menu_name.map(lambda m: unify(m) if isinstance(m, str) else m)
    r = r.merge(cov, on="recipe_id", how="left")
    r[["n_food", "n_priced"]] = r[["n_food", "n_priced"]].fillna(0).astype(int)
    r["all_priced"] = (r.n_food > 0) & (r.n_food == r.n_priced)
    keep = ["recipe_id", "menu_name", "is_menu", "name", "url", "views", "rating", "reviews", "servings", "time", "difficulty", "keywords",
            "n_food", "n_priced", "all_priced", "food_items", "price_items"]
    r[keep].to_parquet(OUT / "recipes.parquet", index=False)

    occ = food.price_source.fillna("미연결").value_counts(normalize=True) * 100
    print(f"레시피 {len(r)}개, 재료 {ing.ingredient.nunique()}종 (양념 {(imap['class'] == '양념').sum()}종)")
    print("식재료 등장 기준 연결:", {k: f"{v:.0f}%" for k, v in occ.items()})
    print(f"식재료가 모두 가격 연결된 레시피: {r.all_priced.sum()}개 ({r.all_priced.mean() * 100:.0f}%), "
          f"80% 이상 연결: {((r.n_priced / r.n_food.clip(lower=1)) >= .8).sum()}개")
    print("연결 안 된 식재료 상위:", ", ".join(f"{a}({b})" for a, b in imap[(imap['class'] == '식재료') & imap.price_item.isna()].head(40)[["ingredient", "n_recipes"]].values))
    print(f"메뉴 {r.is_menu.sum()}개 (메뉴 아님 제외 {(~r.is_menu).sum()}개)")
    print(f"메뉴명 {r.menu_name.notna().sum()}개 ({r.menu_name.notna().mean() * 100:.0f}%), 고유 메뉴 {r.menu_name.nunique()}개. 레시피 많은 메뉴:", r.menu_name.value_counts().head(25).to_dict())


if __name__ == "__main__":
    main()
