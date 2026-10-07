"""원천 데이터 수집 (2016-01-01 ~ 2026-09-30). 조각 단위로 data/raw/<소스>/에 parquet 저장, 이미 받은 조각은 건너뛴다.

    python scripts/collect.py                 # 전체 (가벼운 소스 → 무거운 소스 순)
    python scripts/collect.py at_price asos   # 일부

data.go.kr 하루 호출 한도(aT 10,000 / 축평원 1,000)에 걸리면 해당 소스만 중단하고 다음 소스로 넘어간다.
다음 날 같은 명령을 다시 실행하면 이어서 받는다.
"""
import datetime as dt
import json
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

from common import END, ENV, RAW, START, QuotaExceeded, http_get, http_json

K = ENV["DATA_GO_KR_API_KEY"]
S = dt.date.fromisoformat(START)
E = dt.date.fromisoformat(END)


class KeyRing:
    """data.go.kr 키 여러 개(DATA_GO_KR_API_KEY, DATA_GO_KR_API_KEY_2, ...)를 돌려 쓴다. 호출 한도는 키·API별로 따로 잡힌다."""

    def __init__(self):
        self.keys = [ENV[k] for k in sorted(ENV) if k.startswith("DATA_GO_KR_API_KEY") and ENV[k]]
        self.i = 0

    def xml(self, url, params):
        """기관 서버(축평원 등) XML 응답용. 한도 초과·미등록 키는 다음 키로 넘긴다."""
        while True:
            body = None
            try:
                body = http_get(url, {"serviceKey": self.keys[self.i], **params})
            except QuotaExceeded:
                pass
            if body is not None and "등록되지 않은 서비스키" not in body and "SERVICE_KEY_IS_NOT_REGISTERED" not in body:
                return body
            self.i += 1
            if self.i >= len(self.keys):
                raise QuotaExceeded(url)
            print(f"  - 키 {self.i}번 사용 불가(한도·미등록) → 키 {self.i + 1}번으로 전환", flush=True)

    def json(self, url, params):
        while True:
            try:
                return http_json(url, {"serviceKey": self.keys[self.i], **params}, safe="[]:")
            except QuotaExceeded:
                self.i += 1
                if self.i >= len(self.keys):
                    raise
                print(f"  - 키 {self.i}번 한도 초과 → 키 {self.i + 1}번으로 전환", flush=True)


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype(str).to_parquet(path, index=False)  # 원본은 문자열 그대로 저장, 형 변환은 build 단계에서


def xml_items(text):
    return [{c.tag: c.text for c in it} for it in ET.fromstring(text).iter("item")]


def days(start=S, end=E):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def months(start=S, end=E):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_end(y, m):
    return (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))


# ---------------------------------------------------------------- 가격 (타깃)

def at_price():
    """aT 일별 도·소매 가격: 품목 × 연도 조각."""
    items = pd.read_csv(RAW / "meta" / "items_kamis.csv", dtype=str)
    url = "https://apis.data.go.kr/B552845/perDay/price"
    ring = KeyRing()
    for _, it in items.iterrows():
        for y in range(S.year, E.year + 1):
            path = RAW / "at_price" / f"{it.ctgry_cd}_{it.item_cd}_{y}.parquet"
            if path.exists():
                continue
            rows, page = [], 1
            while True:
                body = ring.json(url, {
                    "returnType": "json", "pageNo": page, "numOfRows": 1000,
                    "cond[exmn_ymd::GTE]": f"{y}0101", "cond[exmn_ymd::LTE]": min(f"{y}1231", E.strftime("%Y%m%d")),
                    "cond[ctgry_cd::EQ]": it.ctgry_cd, "cond[item_cd::EQ]": it.item_cd,
                })["response"]["body"]
                rows += body["items"]["item"] or []
                if page * 1000 >= body["totalCount"]:
                    break
                page += 1
            save(pd.DataFrame(rows), path)
            print(f"  at_price {it.item_nm} {y}: {len(rows)}")


def at_trade():
    """aT 공영도매시장 정산 (거래량): 서울가락(110001) 일 단위 조각, 2018-01부터 제공."""
    url = "https://apis.data.go.kr/B552845/katSale/trades"
    ring = KeyRing()
    for d in days(max(S, dt.date(2018, 1, 1))):
        path = RAW / "at_trade" / f"{d:%Y}" / f"{d}.parquet"
        if path.exists():
            continue
        rows, page = [], 1
        while True:
            body = ring.json(url, {
                "returnType": "json", "pageNo": page, "numOfRows": 50000,  # 이 API는 1,000행 제한이 없어 하루치를 한 번에 받는다
                "cond[trd_clcln_ymd::EQ]": str(d), "cond[whsl_mrkt_cd::EQ]": "110001",
            })["response"]["body"]
            rows += body["items"]["item"] or []
            if page * 50000 >= body["totalCount"]:
                break
            page += 1
        save(pd.DataFrame(rows), path)  # 휴장일은 빈 파일로 남겨 재호출을 막는다
        if d.day == 1:
            print(f"  at_trade {d}: {len(rows)}")


def mafra_wholesale():
    """MAFRA 농수축산물 도매가격: 일 단위 조각 (트래픽 제한 없음)."""
    grid = ENV["MAFRA_GRID_WHOLESALE_PRICE"]
    for d in days():
        path = RAW / "mafra_wholesale" / f"{d:%Y}" / f"{d}.parquet"
        if path.exists():
            continue
        rows, start = [], 1
        while True:
            j = http_json(f"http://211.237.50.150:7080/openapi/{ENV['MAFRA_API_KEY']}/json/{grid}/{start}/{start + 999}",
                          {"EXAMIN_DE": d.strftime("%Y%m%d")})
            g = j.get(grid, {})
            rows += g.get("row", [])
            if start + 999 >= g.get("totalCnt", 0):
                break
            start += 1000
        save(pd.DataFrame(rows), path)
        if d.day == 1:
            print(f"  mafra_wholesale {d}: {len(rows)}")


# ---------------------------------------------------------------- 축산

EKAPE_CONSUMER = {("4301", "22"): ("소", "등심"), ("4304", "27"): ("돼지", "삼겹살")}  # aT 축산 시계열 중 2016~2017년이 필요한 것


def ekape_consumer():
    """aT·KAMIS API는 축산 소매가를 2018년부터만 제공 → 2016~2017년은 축평원 일자별 소비자가격으로 채운다.
    (2018년 이후 aT 값과 원 단위까지 같음을 확인. 응답의 '평년' 행은 버린다.) 하루 1,000회 한도라 월 조각으로 이어받기."""
    U = "http://data.ekape.or.kr/openapi-data/service/user/grade/consumerPriceDaily"
    ring = KeyRing()
    for y, m in months(S, dt.date(2017, 12, 31)):
        for (kind, item), (nm, part) in EKAPE_CONSUMER.items():
            path = RAW / "ekape_consumer" / f"{kind}_{item}_{y}{m:02d}.parquet"
            if path.exists():
                continue
            rows = []
            for d in days(dt.date(y, m, 1), month_end(y, m)):
                if d.weekday() >= 5:
                    continue
                t = ring.xml(U, dict(standYmd=d.strftime("%Y%m%d"), judgeKind=kind, itemCd=item))
                rows += [r for r in xml_items(t) if r.get("standYmd") == d.strftime("%Y%m%d")]
            save(pd.DataFrame(rows), path)
        print(f"  ekape_consumer {y}-{m:02d}")


def mafra_disease():
    path = RAW / "mafra_disease" / "all.parquet"
    if path.exists():
        return
    grid = ENV["MAFRA_GRID_LIVESTOCK_DISEASE"]
    base = f"http://211.237.50.150:7080/openapi/{ENV['MAFRA_API_KEY']}/json/{grid}"
    total = http_json(f"{base}/1/1")[grid]["totalCnt"]
    rows = []
    for s in range(1, total + 1, 1000):
        rows += http_json(f"{base}/{s}/{s + 999}")[grid]["row"]
    save(pd.DataFrame(rows), path)
    print(f"  mafra_disease: {len(rows)}")


def ekape_weekly(start="2015-10-05", end="2024-12-29"):
    """축평원 소 도체 경락: 주(월~일) 단위 조각 (월 단위보다 최신 신호). 주당 1회 호출, 주마다 저장해 이어받기 가능."""
    U = "http://data.ekape.or.kr/openapi-data/service/user/grade/auct/cattle"
    n = 0
    for mon in pd.date_range(start, end, freq="W-MON"):
        sun = mon + pd.Timedelta(days=6)
        path = RAW / "ekape" / "cattle_auction_weekly" / f"{mon:%Y%m%d}.parquet"
        if path.exists():
            continue
        t = http_get(U, dict(serviceKey=K, startYmd=f"{mon:%Y%m%d}", endYmd=f"{sun:%Y%m%d}", numOfRows=1000, pageNo=1))
        save(pd.DataFrame(xml_items(t)).assign(week_start=f"{mon:%Y%m%d}"), path)
        n += 1
        if n % 50 == 0:
            print(f"  ekape_weekly {mon:%Y-%m-%d} ({n}회)", flush=True)
    print(f"  ekape_weekly 완료: 이번 실행 {n}회 호출")


def ekape():
    """축평원: 월 단위 조각 (기간 조회 시 월 합계로 응답). 하루 1,000회 한도."""
    U = "http://data.ekape.or.kr/openapi-data/service/user"
    for y, m in months():
        s, e = dt.date(y, m, 1), min(month_end(y, m), E)
        for name, op in [("cattle_auction", "grade/auct/cattle"), ("pig_auction", "grade/auct/pigGrade"),
                         ("pig_rep_price", "grade/auct/pigRepresentativePrice")]:
            path = RAW / "ekape" / name / f"{y}{m:02d}.parquet"
            if path.exists():
                continue
            t = http_get(f"{U}/{op}", dict(serviceKey=K, startYmd=f"{s:%Y%m%d}", endYmd=f"{e:%Y%m%d}", numOfRows=1000, pageNo=1))
            save(pd.DataFrame(xml_items(t)).assign(ym=f"{y}{m:02d}"), path)
        for kind in ("4301", "4304"):
            path = RAW / "ekape" / "stock" / f"{y}{m:02d}_{kind}.parquet"
            if path.exists():
                continue
            t = http_get(f"{U}/grade/LPStock", dict(serviceKey=K, standYyyy=y, standMm=f"{m:02d}", judgeKind=kind))
            save(pd.DataFrame(xml_items(t)).assign(ym=f"{y}{m:02d}"), path)
        if m == 1:
            print(f"  ekape {y}")


# ---------------------------------------------------------------- 수입

HS_CHAPTERS = ["02", "03", "04", "07", "08", "09", "10", "12", "16", "25"]  # 육류, 어류, 낙농·난, 채소, 과일, 향신료(고추·생강), 곡물, 유지종자, 육·어 조제품, 소금


def customs():
    for ch in HS_CHAPTERS:
        for y in range(S.year, E.year + 1):
            path = RAW / "customs" / f"{ch}_{y}.parquet"
            if path.exists():
                continue
            t = http_get("https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList",
                         dict(serviceKey=K, strtYymm=f"{y}01", endYymm=f"{y}12" if y < E.year else f"{E:%Y%m}", hsSgn=ch))
            save(pd.DataFrame(xml_items(t)), path)
            print(f"  customs {ch} {y}")


# ---------------------------------------------------------------- 기상·수산 환경

def asos():
    """ASOS 일자료: 전 관측소. 지점 목록은 첫 실행 때 90~300번을 조회해 만든다."""
    url = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    stn_path = RAW / "meta" / "asos_stations.csv"
    if not stn_path.exists():
        found = []
        for sid in range(90, 300):
            j = json.loads(http_get(url, dict(serviceKey=K, dataType="JSON", dataCd="ASOS", dateCd="DAY",
                                               startDt="20260901", endDt="20260901", stnIds=sid, numOfRows=1, pageNo=1)))
            items = j.get("response", {}).get("body", {}).get("items", {})
            if items and items.get("item"):
                found.append((sid, items["item"][0]["stnNm"]))
        pd.DataFrame(found, columns=["stnId", "stnNm"]).to_csv(stn_path, index=False, encoding="utf-8-sig")
        print(f"  asos 지점 {len(found)}개")
    last = min(E, dt.date.today() - dt.timedelta(days=1))
    for sid in pd.read_csv(stn_path).stnId:
        for y in range(S.year, last.year + 1):
            path = RAW / "asos" / f"{sid}_{y}.parquet"
            if path.exists():
                continue
            s, e = dt.date(y, 1, 1), min(dt.date(y, 12, 31), last)
            j = json.loads(http_get(url, dict(serviceKey=K, dataType="JSON", dataCd="ASOS", dateCd="DAY",
                                               startDt=f"{s:%Y%m%d}", endDt=f"{e:%Y%m%d}", stnIds=sid, numOfRows=999, pageNo=1)))
            items = j.get("response", {}).get("body", {}).get("items", {}) or {}
            save(pd.DataFrame(items.get("item", [])), path)
        print(f"  asos {sid}")


def nifs():
    N = "https://www.nifs.go.kr/OpenAPI_json"
    for y in range(S.year, E.year + 1):
        s, e = f"{y}0101", min(f"{y}1231", E.strftime("%Y%m%d"))
        for name, id_, key in [("coast_temp", "cooList", "NIFS_COO_API_KEY"), ("redtide", "redtideList", "NIFS_REDTIDE_API_KEY"),
                               ("jellyfish", "jellyList", "NIFS_JELLY_API_KEY"), ("farm_env", "femoSeaList", "NIFS_FEMO_API_KEY")]:   # farm_env: 어장환경(양식장 수질)
            path = RAW / "nifs" / name / f"{y}.parquet"
            if path.exists():
                continue
            j = http_json(N, dict(id=id_, key=ENV[key], sdate=s, edate=e))
            items = j.get("body", {}).get("item", []) or []
            if name == "redtide":  # 해역별 상세(item2)를 펼친다
                items = [{**{k: v for k, v in it.items() if k != "item2"}, **sub} for it in items for sub in (it.get("item2") or [{}])]
            save(pd.DataFrame(items), path)
        print(f"  nifs {y}")


# ---------------------------------------------------------------- 경제·통계

def ecos():
    key = ENV["ECOS_API_KEY"]

    def search(stat, cycle, start, end, item):
        rows, s = [], 1
        while True:
            j = http_json(f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{s}/{s + 9999}/{stat}/{cycle}/{start}/{end}/{item}")
            r = j.get("StatisticSearch", {})
            rows += r.get("row", [])
            if s + 9999 >= r.get("list_total_count", 0):
                return rows
            s += 10000

    def items_of(stat):
        j = http_json(f"https://ecos.bok.or.kr/api/StatisticItemList/{key}/json/kr/1/5000/{stat}")
        return j["StatisticItemList"]["row"]

    targets = {
        "fx_usd": [("731Y001", "D", f"{S:%Y%m%d}", f"{E:%Y%m%d}", "0000001")],
        # 소비자물가: 총지수 + 식료품(A01*) 품목
        "cpi": [("901Y009", "M", f"{S:%Y%m}", f"{E:%Y%m}", c) for c in
                ["0"] + sorted({r["ITEM_CODE"] for r in items_of("901Y009") if r["ITEM_CODE"].startswith("A01") and r["CYCLE"] == "M"})],
        # 수입물가: 농림수산품(101*) + 식료품(3011*)
        "import_price": [("401Y015", "M", f"{S:%Y%m}", f"{E:%Y%m}", c) for c in
                         sorted({r["ITEM_CODE"] for r in items_of("401Y015") if r["ITEM_CODE"].startswith(("101", "3011")) and r["CYCLE"] == "M"})],
        # 국제상품가격: 원유·곡물 등 전체
        "intl_commodity": [("902Y003", "M", f"{S:%Y%m}", f"{E:%Y%m}", c) for c in
                           sorted({r["ITEM_CODE"] for r in items_of("902Y003") if r["CYCLE"] == "M"})],
    }
    for name, reqs in targets.items():
        path = RAW / "ecos" / f"{name}.parquet"
        if path.exists():
            continue
        rows = [r for q in reqs for r in search(*q)]
        save(pd.DataFrame(rows), path)
        print(f"  ecos {name}: {len(rows)}")


KOSIS_TABLES = {
    # 이름: (orgId, tblId, prdSe, 고정 objL 목록, itmId)
    "livestock_census": ("101", "DT_1EO200", "Q", ["ALL"], "ALL"),                       # 가축동향: 소 (분기)
    "pig_census_old": ("101", "DT_1EO051", "Q", ["ALL", "00"], "ALL"),                  # 가축동향: 돼지 (~2017 1분기)
    "pig_census": ("101", "DT_1EO311", "Q", ["ALL", "00"], "ALL"),                      # 가축동향: 돼지 (2017 1분기~)
    "chicken_census": ("101", "DT_1EO071", "Q", ["ALL", "00"], "ALL"),                  # 가축동향: 산란계·육용계
    "fishery_production": ("101", "DT_1EW0004", "M", ["0", "ALL", "00", "0"], "T01+T05"),  # 어업생산 품종별 (월): 생산량·금액
    "veg_leaf": ("101", "DT_1ET0028", "Y", ["ALL"], "ALL"),                               # 엽채류 생산 (연, 시도별 - 주산지 가중치용)
    "veg_root": ("101", "DT_1ET0029", "Y", ["ALL"], "ALL"),                                # 근채류
    "veg_seasoning": ("101", "DT_1ET0291", "Y", ["ALL"], "ALL"),                           # 조미채소
    "food_crops": ("101", "DT_1ET0021", "Y", ["ALL"], "ALL"),                             # 식량작물
    "fruit": ("101", "DT_1ET0292", "Y", ["ALL"], "ALL"),                                   # 과실
    "fuel_price": ("318", "TX_31802_A000", "M", ["ALL"], "ALL"),                          # 주유소 평균 판매가격 (월)
}


def kosis():
    key = ENV["KOSIS_API_KEY"]
    for name, (org, tbl, prd, objs, itm) in KOSIS_TABLES.items():
        fmt = {"Y": "%Y", "Q": "%Y", "M": "%Y%m"}[prd]
        for y in range(S.year, E.year + 1):  # 4만 셀 제한 때문에 연 단위로 나눈다
            path = RAW / "kosis" / name / f"{y}.parquet"
            if path.exists():
                continue
            s, e = dt.date(y, 1, 1), min(dt.date(y, 12, 31), E)
            p = dict(method="getList", apiKey=key, orgId=org, tblId=tbl, itmId=itm, prdSe=prd, format="json", jsonVD="Y",
                     startPrdDe=s.strftime(fmt) + ("01" if prd == "Q" else ""), endPrdDe=e.strftime(fmt) + ("04" if prd == "Q" else ""))
            p.update({f"objL{i + 1}": v for i, v in enumerate(objs)})
            j = json.loads(http_get("https://kosis.kr/openapi/Param/statisticsParameterData.do", p))
            if isinstance(j, dict) and j.get("err"):
                if j["err"] == "30":  # 데이터 없음 (해당 연도 미공표)
                    j = []
                else:
                    raise RuntimeError(f"KOSIS {tbl} {y}: {j}")
            save(pd.DataFrame(j), path)
        print(f"  kosis {name}")


def recipe():
    """MAFRA 레시피 기본정보·재료정보 (후순위: 메뉴 추천·조리 특성용)."""
    for name, grid_key in [("basic", "MAFRA_GRID_RECIPE_BASIC"), ("ingredient", "MAFRA_GRID_RECIPE_INGREDIENT")]:
        path = RAW / "recipe" / f"{name}.parquet"
        if path.exists():
            continue
        grid = ENV[grid_key]
        base = f"http://211.237.50.150:7080/openapi/{ENV['MAFRA_API_KEY']}/json/{grid}"
        total = http_json(f"{base}/1/1")[grid]["totalCnt"]
        rows = [r for s in range(1, total + 1, 1000) for r in http_json(f"{base}/{s}/{s + 999}")[grid]["row"]]
        save(pd.DataFrame(rows), path)
        print(f"  recipe {name}: {len(rows)}")


def foodsafety_recipe():
    """식약처 식품안전나라 조리식품의 레시피 DB (COOKRCP01): 재료 g 분량·영양정보 포함, 1회 최대 1,000건."""
    path = RAW / "recipe" / "foodsafety_cookrcp01.parquet"
    if path.exists():
        return
    base = f"http://openapi.foodsafetykorea.go.kr/api/{ENV['FOODSAFETY_API_KEY']}/COOKRCP01/json"
    total = int(http_json(f"{base}/1/1")["COOKRCP01"]["total_count"])
    rows = [r for s in range(1, total + 1, 1000) for r in http_json(f"{base}/{s}/{min(s + 999, total)}")["COOKRCP01"]["row"]]
    save(pd.DataFrame(rows), path)
    print(f"  foodsafety_recipe: {len(rows)}")


def holidays_kr():
    import holidays
    path = RAW / "calendar" / "holidays.parquet"
    rows = [(d, n) for d, n in sorted(holidays.KR(years=range(S.year, E.year + 2)).items())]
    save(pd.DataFrame(rows, columns=["date", "name"]), path)
    print(f"  holidays: {len(rows)}")


SOURCES = {  # 가벼운 것부터
    "holidays": holidays_kr, "ecos": ecos, "kosis": kosis, "customs": customs, "nifs": nifs,
    "mafra_disease": mafra_disease, "recipe": recipe, "foodsafety_recipe": foodsafety_recipe, "ekape": ekape, "ekape_weekly": ekape_weekly, "ekape_consumer": ekape_consumer, "asos": asos,
    "mafra_wholesale": mafra_wholesale, "at_price": at_price, "at_trade": at_trade,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    for n in names:
        print(f"[{n}]", flush=True)
        try:
            SOURCES[n]()
        except QuotaExceeded as q:
            print(f"  ! 호출 한도 초과 ({q}) - 내일 다시 실행하면 이어서 받음", flush=True)
        except Exception as e:
            print(f"  ! 실패: {e!r}", flush=True)
    print("done", flush=True)
