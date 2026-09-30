"""전국통합식품영양성분정보 표준데이터 다운로드.

data.go.kr 표준데이터 페이지의 '다운로드' 버튼이 호출하는 엔드포인트를 그대로 사용한다.
    python scripts/download_nutrition.py            # 전체
    python scripts/download_nutrition.py food       # 하나만
"""
import csv
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# 이름: (publicDataPk, 저장 파일명)
DATASETS = {
    "raw_material": ("15100065", "nutrition_raw_material.csv"),  # 원재료성식품
    "food": ("15100070", "nutrition_food.csv"),                   # 음식(완성 요리)
}
BASE = "https://www.data.go.kr"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nutrition"
PER_PAGE = 10000


def get_json(pk, path, params):
    # colNmList는 같은 키를 반복해서 보내야 함 (jQuery traditional: true)
    url = f"{BASE}{path}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": f"{BASE}/data/{pk}/standard.do"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def download(pk, filename):
    header = get_json(pk, "/download/columList.json", {"pk": pk, "ext": "CSV"})
    cols_en = [c["columCode"] for c in header["columList"]]
    cols_kr = [c["columNm"] for c in header["columList"]]
    total = header["totalCount"]
    print(f"{header['fileName']}: {total}행, {len(cols_en)}컬럼")

    rows = []
    for page in range(1, math.ceil(total / PER_PAGE) + 1):
        data = get_json(pk, "/download/standard.json", {
            "colNmList": header["tableVO"]["colNmList"],
            "totalCount": total,
            "svcTableNm": header["tableVO"]["svcTableNm"],
            "perPage": PER_PAGE,
            "page": page,
        })
        rows.extend(data)
        print(f"  page {page}: 누적 {len(rows)}행")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / filename
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols_kr)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols_en])
    print(f"저장: {out}")
    if len(rows) != total:
        sys.exit(f"경고: 기대 {total}행, 실제 {len(rows)}행")


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        download(*DATASETS[name])
