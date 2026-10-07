"""생필품(가공식품) 가격: 한국소비자원 생필품 가격 정보 API → data/raw/price_go/, data/reference/price_go_goods.csv

    python scripts/collect_price_go.py map        # 최초 1회: 상품 ID ↔ 상품명 매칭표 만들기 (월별 CSV와 같은 조사일 이용)
    python scripts/collect_price_go.py update     # 갱신: 최신 조사일의 전체 상품 가격 받기 (조사는 약 2주마다)

원칙: 생필품은 예측하지 않는다. 최신 조사 가격이 다음 조사 때까지 유지된다고 보고, 새 조사가 올라오면 갱신만 한다.
조회는 매장 단위(호출 1회 = 매장 1곳의 전 상품, 조사일당 약 270회). API는 상품 ID·매장 ID·가격만 주고 상품명이 없다 → 같은 조사일의 월별 CSV(상품명·매장명 포함)와
상품별 '매장 가격 분포'를 비교해 1:1로 짝짓는다 (헝가리안 알고리즘).
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment

from common import ENV, RAW, ROOT

URL = "https://apis.data.go.kr/B551919/ProductPriceInfoService/getProductPriceInfoSvc"
OUT = RAW / "price_go"
MAP = ROOT / "data" / "reference" / "price_go_goods.csv"
FIELDS = ["goodInspectDay", "entpId", "goodId", "goodPrice", "plusoneYn", "goodDcYn"]
S = requests.Session()


def query(**params):
    for _ in range(3):
        try:
            t = S.get(URL, params={"serviceKey": ENV["DATA_GO_KR_API_KEY"], **params}, timeout=30).text
            if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in t:
                raise SystemExit("하루 호출 한도 초과 — 내일 다시 실행하면 이어서 받음")
            items = re.findall(r"<iros\.openapi\.service\.vo\.goodPriceVO>(.*?)</iros", t, re.S)
            return [{f: (re.search(f"<{f}>(.*?)</{f}>", it) or [None, None])[1] for f in FIELDS} for it in items]
        except requests.RequestException:
            continue
    return []


def stores(day, probe_ids=(14, 11, 120, 143, 174, 201)):
    """그 조사일에 조사된 매장 ID 목록: 흔한 상품 몇 개를 조회해 매장 ID를 모은다 (호출 6회)."""
    ents = set()
    for g in probe_ids:
        ents |= {r["entpId"] for r in query(goodInspectDay=day, goodId=str(g))}
    return sorted(ents, key=int)


def prices(day):
    """조사일 전체 가격: 매장 단위로 조회 (매장 1곳 = 호출 1회로 그 매장의 전 상품). 중간에 끊겨도 이어받기."""
    part = OUT / f"_part_{day}.parquet"
    done = pd.read_parquet(part) if part.exists() else pd.DataFrame(columns=FIELDS)
    todo = [e for e in stores(day) if e not in set(done.entpId)]
    print(f"    {day}: 매장 {len(todo)}곳 조회 (완료 {done.entpId.nunique()}곳)", flush=True)
    buf = []
    try:
        for n, e in enumerate(todo, 1):
            buf += query(goodInspectDay=day, entpId=e)
            if n % 50 == 0:
                print(f"    {day}: {n}/{len(todo)}", flush=True)
    finally:
        done = pd.concat([done, pd.DataFrame(buf, columns=FIELDS)], ignore_index=True)
        done.to_parquet(part, index=False)
    done["goodPrice"] = pd.to_numeric(done.goodPrice)
    part.unlink()
    return done


def is_survey_day(day, probe_ids):
    return any(query(goodInspectDay=day, goodId=str(g)) for g in probe_ids)


def build_map():
    csv = pd.read_csv(next(OUT.glob("생필품_가격_*.csv")), encoding="cp949")
    days = sorted(csv.조사일.unique())                                       # 예: 2026-08-07, 2026-08-28
    api = {}
    for i, d in enumerate(days):
        dd = d.replace("-", "")
        path = OUT / f"api_{dd}.parquet"
        if path.exists():
            api[d] = pd.read_parquet(path)
            continue
        api[d] = prices(dd)
        api[d].to_parquet(path, index=False)
    # 상품별 매장 가격 분포(가격 → 매장 수) 비교: 두 조사일 합산 다중집합 겹침(Jaccard)
    def dist(df, key, val):
        return {k: Counter(g[val].astype(int)) for k, g in df.groupby(key)}
    a = [dist(api[d], "goodId", "goodPrice") for d in days]
    c = [dist(csv[csv.조사일 == d], "상품명", "판매가격") for d in days]
    gids, names = sorted(set().union(*a)), sorted(set().union(*c))
    sim = np.zeros((len(gids), len(names)))
    for t in range(len(days)):
        for i, g in enumerate(gids):
            ag = a[t].get(g)
            if not ag:
                continue
            for j, nm in enumerate(names):
                cn = c[t].get(nm)
                if cn:
                    inter = sum((ag & cn).values())
                    if inter:
                        sim[i, j] += inter / sum((ag | cn).values())
    sim /= len(days)
    r, k = linear_sum_assignment(-sim)
    m = pd.DataFrame({"goodId": [gids[i] for i in r], "상품명": [names[j] for j in k], "match_score": sim[r, k].round(3)})
    m = m[m.match_score > 0].sort_values("match_score", ascending=False)
    m["confident"] = m.match_score >= 0.5
    m.to_csv(MAP, index=False, encoding="utf-8-sig")
    print(f"매칭 {len(m)}개 (API 상품 {len(gids)}, CSV 상품 {len(names)}), 확신(≥0.5) {m.confident.sum()}개, "
          f"점수 중앙값 {m.match_score.median():.2f}")
    print("낮은 점수 예:", m.tail(8).to_string(index=False))


def update():
    ids = [14, 11, 120, 143, 174]
    have = sorted(OUT.glob("api_*.parquet"))
    last = pd.Timestamp(have[-1].stem[4:]) if have else pd.Timestamp("2026-08-28")
    new = [d for d in pd.date_range(last + pd.Timedelta(days=7), pd.Timestamp.today(), freq="W-FRI")
           if is_survey_day(d.strftime("%Y%m%d"), ids[:5])]
    if not new:
        print(f"새 조사일 없음 (최신 {last:%Y-%m-%d})")
        return
    for d in new:
        df = prices(d.strftime("%Y%m%d"))
        df.to_parquet(OUT / f"api_{d:%Y%m%d}.parquet", index=False)
        print(f"  {d:%Y-%m-%d}: {len(df):,}행, 상품 {df.goodId.nunique()}개")


if __name__ == "__main__":
    {"map": build_map, "update": update}[sys.argv[1] if len(sys.argv) > 1 else "update"]()
