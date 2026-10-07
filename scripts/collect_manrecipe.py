"""만개의레시피 자취·간단 요리 메뉴 수집 (비공식 출처) → data/raw/manrecipe/

    python scripts/collect_manrecipe.py

수집 원칙 (이용약관 제11·15조: 레시피 저작권은 작성자, 영리 이용·제3자 제공 금지):
- 메뉴명, 재료(이름·분량), 인분·조리시간·난이도, 조회수·평점, 작성일, 원문 URL만 받는다. 조리 과정 본문과 사진은 받지 않는다.
- 앱에는 메뉴명과 원문 링크만 노출한다. 수집 파일은 저장소·외부에 공유하지 않는다 (data/raw는 gitignore).
- 요청 사이 1.2초 이상 간격. 중간에 끊겨도 다시 실행하면 이어서 받는다.
"""
import json
import re
import time
from urllib.parse import quote

import pandas as pd
import requests

from common import RAW

OUT = RAW / "manrecipe"
BASE = "https://www.10000recipe.com"
KEYWORDS = ["자취", "자취요리", "간단요리", "초간단", "1인분", "혼밥", "간단 반찬", "원룸"]
MAX_PAGES = 15                          # 검색어당 최대 15페이지 (페이지당 약 35개)
DELAY = 1.2
HEAD = {"User-Agent": "Mozilla/5.0 (student ML project, non-commercial)"}
S = requests.Session()
S.headers.update(HEAD)


def get(url):
    for i in range(3):
        try:
            r = S.get(url, timeout=30)
            time.sleep(DELAY)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            time.sleep(5 * (i + 1))
    return None


def list_ids():
    path = OUT / "ids.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"recipe_id": str})
    rows = []
    for kw in KEYWORDS:
        for page in range(1, MAX_PAGES + 1):
            h = get(f"{BASE}/recipe/list.html?q={quote(kw)}&order=reco&page={page}")
            ids = list(dict.fromkeys(re.findall(r'href="/recipe/(\d+)" class="common_sp_link"', h or "")))
            if not ids:
                break
            rows += [(i, kw, page) for i in ids]
        print(f"  목록 {kw}: 누적 {len(set(r[0] for r in rows))}개", flush=True)
    df = pd.DataFrame(rows, columns=["recipe_id", "keyword", "page"]).groupby("recipe_id", as_index=False).agg(
        keywords=("keyword", lambda s: ";".join(dict.fromkeys(s))), first_page=("page", "min"))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def num(s):
    m = re.search(r"[\d,]+", s or "")
    return int(m.group().replace(",", "")) if m else None


def parse(rid, h):
    ld = {}
    for b in re.findall(r'<script type="application/ld\+json">(.*?)</script>', h, re.S):
        try:
            d = json.loads(b, strict=False)
            if d.get("@type") == "Recipe":
                ld = d
        except ValueError:
            pass
    info = {k: (re.search(k + r'">([^<]*)', h) or [None, None])[1] for k in ("view2_summary_info1", "view2_summary_info2", "view2_summary_info3")}
    ing = [(a.strip(), b.strip()) for a, b in re.findall(
        r'<div class="ingre_list_name">\s*<a[^>]*>\s*([^<]+?)\s*</a>.*?<span class="ingre_list_ea">([^<]*)</span>', h, re.S)]
    if not ing:                                                     # 재료 영역 구조가 다른 레시피는 JSON-LD 재료 문자열 사용
        ing = [(x.strip(), "") for x in ld.get("recipeIngredient", [])]
    ing_text = None
    if not ing:                                                     # 예전 형식: 재료가 본문 문장에만 있음 → "재료" ~ "만들기" 구간만 보관
        desc = re.search(r'<meta name="description" content="([^"]*)"', h)
        m = re.search(r"재료[^:：]*[:：]\s*(.+?)(?:■\s*만들|만드는\s*법|만들기|조리\s*방법|how to|\d\s*\.\.\.)", desc.group(1) if desc else "", re.S | re.I)
        ing_text = m.group(1).strip()[:500] if m else None
    hit = re.search(r'class="hit font_num">\s*([\d,]+)', h) or re.search(r'조회수\s*([\d,]+)', h)
    rating = ld.get("aggregateRating") or {}
    if not rating:
        pt, ea = re.search(r'class="hit_rv_pt">([\d.]+)', h), re.search(r'class="hit_rv_ea">\((\d+)\)', h)
        rating = {"ratingValue": pt and pt.group(1), "reviewCount": ea and ea.group(1)}
    return {"recipe_id": rid, "name": (ld.get("name") or "").strip(), "url": f"{BASE}/recipe/{rid}",
            "servings": info["view2_summary_info1"], "time": info["view2_summary_info2"], "difficulty": info["view2_summary_info3"],
            "views": num(hit.group(1)) if hit else None, "rating": rating.get("ratingValue"), "reviews": rating.get("reviewCount"),
            "published": ld.get("datePublished"), "n_ingredients": len(ing),
            "ingredients": json.dumps(ing, ensure_ascii=False), "ingredients_text": ing_text}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = list_ids()
    done_path = OUT / "recipes.parquet"
    done = pd.read_parquet(done_path) if done_path.exists() else pd.DataFrame()
    have = set(done.recipe_id) if len(done) else set()
    todo = [i for i in ids.recipe_id if i not in have]
    print(f"레시피 {len(ids)}개 중 남은 {len(todo)}개 (예상 {len(todo) * DELAY / 60:.0f}분)", flush=True)
    buf = []
    for n, rid in enumerate(todo, 1):
        h = get(f"{BASE}/recipe/{rid}")
        if h:
            buf.append(parse(rid, h))
        if len(buf) >= 100 or n == len(todo):
            done = pd.concat([done, pd.DataFrame(buf)], ignore_index=True)
            done.to_parquet(done_path, index=False)
            buf = []
            print(f"  상세 {len(done)}개 저장", flush=True)
    if "ingredients_text" not in done:
        done["ingredients_text"] = None
    redo = done.index[(done.n_ingredients == 0) & done.ingredients_text.isna()]
    if len(redo):                                                   # 재료를 못 읽은 예전 형식 레시피만 다시 받아 재료 문장 보관
        print(f"재료 문장 보완 {len(redo)}개 (예상 {len(redo) * DELAY / 60:.0f}분)", flush=True)
        for i in redo:
            h = get(done.at[i, "url"])
            if h:
                done.at[i, "ingredients_text"] = parse(done.at[i, "recipe_id"], h)["ingredients_text"]
        done.to_parquet(done_path, index=False)
        print(f"  보완 완료: 재료 문장 확보 {done.loc[redo, 'ingredients_text'].notna().sum()}개", flush=True)
    done = done.merge(ids[["recipe_id", "keywords"]], on="recipe_id", how="left")
    print(done[["servings", "time", "difficulty"]].apply(lambda s: s.value_counts().head(6).to_dict()).to_string())


if __name__ == "__main__":
    main()
