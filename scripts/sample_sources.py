"""소스별 샘플 응답을 data/raw/samples/에 저장한다 (컬럼·기간 확인용).

    python scripts/sample_sources.py            # 전체
    python scripts/sample_sources.py KAMIS ECOS # 일부
"""
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "samples"


def load_env():
    env = {}
    for line in open(ROOT / ".env", encoding="utf-8"):
        m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
        if m:
            env[m[1]] = m[2].strip().strip('"').strip("'")
    return env


E = load_env()
K = E["DATA_GO_KR_API_KEY"]
MAFRA = f"http://211.237.50.150:7080/openapi/{E['MAFRA_API_KEY']}/json"
EKAPE = "http://data.ekape.or.kr/openapi-data/service/user"
KAMIS = dict(p_cert_key=E["KAMIS_CERT_KEY"], p_cert_id=E["KAMIS_CERT_ID"], p_returntype="json")
ECOS = f"https://ecos.bok.or.kr/api/StatisticSearch/{E['ECOS_API_KEY']}/json/kr/1/100"
ECOS_ITEMS = f"https://ecos.bok.or.kr/api/StatisticItemList/{E['ECOS_API_KEY']}/json/kr/1/2000"
KOSIS = dict(apiKey=E["KOSIS_API_KEY"], format="json", jsonVD="Y")
NIFS = "https://www.nifs.go.kr/OpenAPI_json"

# (소스, 이름, url, params)
SAMPLES = [
    # KAMIS: 부류코드 100 식량 200 채소 300 특용 400 과일 500 축산 600 수산 / 01 소매 02 도매
    *[("KAMIS", f"daily_retail_{c}", "http://www.kamis.or.kr/service/price/xml.do",
       dict(action="dailyPriceByCategoryList", p_product_cls_code="01", p_country_code="", p_regday="2026-09-22",
            p_convert_kg_yn="N", p_item_category_code=str(c), **KAMIS)) for c in (100, 200, 300, 400, 500, 600)],
    *[("KAMIS", f"daily_wholesale_{c}", "http://www.kamis.or.kr/service/price/xml.do",
       dict(action="dailyPriceByCategoryList", p_product_cls_code="02", p_country_code="", p_regday="2026-09-22",
            p_convert_kg_yn="N", p_item_category_code=str(c), **KAMIS)) for c in (100, 200, 300, 400, 500, 600)],
    ("KAMIS", "period_cabbage_retail_2000", "http://www.kamis.or.kr/service/price/xml.do",
     dict(action="periodProductList", p_productclscode="01", p_startday="2000-01-01", p_endday="2000-01-31",
          p_itemcategorycode="200", p_itemcode="211", p_kindcode="01", p_productrankcode="04", p_countrycode="1101",
          p_convert_kg_yn="N", **KAMIS)),
    ("KAMIS", "period_cabbage_retail_2026", "http://www.kamis.or.kr/service/price/xml.do",
     dict(action="periodProductList", p_productclscode="01", p_startday="2026-09-01", p_endday="2026-09-29",
          p_itemcategorycode="200", p_itemcode="211", p_kindcode="01", p_productrankcode="04", p_countrycode="1101",
          p_convert_kg_yn="N", **KAMIS)),
    ("KAMIS", "period_mackerel_retail_2026", "http://www.kamis.or.kr/service/price/xml.do",
     dict(action="periodProductList", p_productclscode="01", p_startday="2026-09-01", p_endday="2026-09-29",
          p_itemcategorycode="600", p_itemcode="611", p_kindcode="05", p_productrankcode="04", p_countrycode="1101",
          p_convert_kg_yn="N", **KAMIS)),
    # data.go.kr
    ("ASOS", "daily", "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList",
     dict(serviceKey=K, dataType="JSON", dataCd="ASOS", dateCd="DAY", startDt="20260901", endDt="20260902", stnIds="108", numOfRows=2, pageNo=1)),
    ("EKAPE", "consumer_price_daily_beef", f"{EKAPE}/grade/consumerPriceDaily",
     dict(serviceKey=K, standYmd="20260922", judgeKind="4301", itemCd="21")),
    ("EKAPE", "consumer_price_daily_pork", f"{EKAPE}/grade/consumerPriceDaily",
     dict(serviceKey=K, standYmd="20260922", judgeKind="4304", itemCd="31")),
    ("EKAPE", "stock", f"{EKAPE}/grade/LPStock", dict(serviceKey=K, issueNo="", numOfRows=5, pageNo=1)),
    ("EKAPE", "pig_representative_price", f"{EKAPE}/grade/auct/pigRepresentativePrice",
     dict(serviceKey=K, startYmd="20260901", endYmd="20260922")),
    ("EKAPE", "cattle_auction", f"{EKAPE}/grade/auct/cattle", dict(serviceKey=K, startYmd="20260922", endYmd="20260922")),
    ("EKAPE", "pig_grade_auction", f"{EKAPE}/grade/auct/pigGrade", dict(serviceKey=K, startYmd="20260922", endYmd="20260922")),
    ("CUSTOMS", "item_country_trade_0303", "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList",
     dict(serviceKey=K, strtYymm="202607", endYymm="202607", hsSgn="0303")),
    ("CUSTOMS", "item_trade_0203", "https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList",
     dict(serviceKey=K, strtYymm="202607", endYymm="202607", hsSgn="0203")),
    ("SUHYUP", "market_info", "https://apis.data.go.kr/1192000/select0020List/getselect0020List",
     dict(serviceKey=K, type="json", pageNo=1, numOfRows=3)),
    # MAFRA
    ("MAFRA", "wholesale_price", f"{MAFRA}/{E['MAFRA_GRID_WHOLESALE_PRICE']}/1/1000", dict(EXAMIN_DE="20260922")),
    ("MAFRA", "livestock_disease_head", f"{MAFRA}/{E['MAFRA_GRID_LIVESTOCK_DISEASE']}/1/3", None),
    ("MAFRA", "livestock_disease_tail", f"{MAFRA}/{E['MAFRA_GRID_LIVESTOCK_DISEASE']}/46170/46172", None),
    ("MAFRA", "veg_production", f"{MAFRA}/{E['MAFRA_GRID_VEG_PRODUCTION']}/1/3", None),
    # ECOS: 원/달러 환율, 소비자물가 품목별, 수입물가
    ("ECOS", "usd_krw_daily", f"{ECOS}/731Y001/D/20260901/20260929/0000001", None),
    ("ECOS", "items_cpi_901Y009", f"{ECOS_ITEMS}/901Y009", None),
    ("ECOS", "items_import_price_401Y015", f"{ECOS_ITEMS}/401Y015", None),
    # KOSIS 검색
    *[("KOSIS", f"search_{w}", "https://kosis.kr/openapi/statisticsSearch.do",
       dict(method="getList", searchNm=w, startCount=1, resultCount=20, sort="RANK", **KOSIS))
      for w in ("가축동향", "농작물생산조사", "어업생산동향", "농산물생산비", "양곡소비량")],
    # NIFS
    ("NIFS", "risa_list", NIFS, dict(id="risaList", key=E["NIFS_RISA_API_KEY"])),
    ("NIFS", "coo_list", NIFS, dict(id="cooList", key=E["NIFS_COO_API_KEY"], sdate="20260801", edate="20260803")),
    ("NIFS", "femo_sea", NIFS, dict(id="femoSeaList", key=E["NIFS_FEMO_API_KEY"], sdate="20260801", edate="20260831")),
    # OPINET
    ("OPINET", "avg_recent", "https://www.opinet.co.kr/api/avgRecentPrice.do", dict(out="json", code=E["OPINET_API_KEY"])),
]


def fetch(url, params):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: " + e.read().decode("utf-8", "replace")
    except Exception as e:
        return f"ERROR: {e!r}"


def main():
    only = set(sys.argv[1:])
    secrets = [v for k, v in E.items() if len(v) >= 6 and "_GRID_" not in k]
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name, url, params in SAMPLES:
        if only and src not in only:
            continue
        body = fetch(url, params)
        for s in secrets:
            body = body.replace(s, "***")
        ext = "xml" if body.lstrip().startswith("<") else "json"
        path = OUT / f"{src}_{name}.{ext}"
        path.write_text(body, encoding="utf-8")
        print(f"{src:8} {name:32} {len(body):>8}B  {re.sub(r'\s+', ' ', body[:90])}")


if __name__ == "__main__":
    main()
