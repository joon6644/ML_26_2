""".env의 API 키가 실제로 동작하는지 사이트/API별로 샘플 호출해서 확인한다.

    python scripts/check_keys.py
키 값은 출력하지 않는다.
"""
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    for line in open(ROOT / ".env", encoding="utf-8"):
        m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
        if m:
            env[m[1]] = m[2].strip().strip('"').strip("'")
    return env


ENV = load_env()
SECRETS = [v for k, v in ENV.items() if len(v) >= 6 and "_GRID_" not in k]


def mask(s):
    for v in SECRETS:
        s = s.replace(v, "***").replace(urllib.parse.quote(v, safe=""), "***")
    return s


def call(url, params=None):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, repr(e)


# (판정 규칙) 본문에 ok 패턴이 있으면 성공, fail 패턴이 있으면 실패
FAIL = r"SERVICE_KEY_IS_NOT_REGISTERED|SERVICE ACCESS DENIED|NO_OPENAPI_SERVICE|INVALID_REQUEST|인증키|인증 ?실패|ERROR-|\"data\":\[\"900\"\]|err_msg|Unauthorized|LIMITED_NUMBER"

K = ENV.get("DATA_GO_KR_API_KEY", "")
TESTS = [
    # --- data.go.kr (키 하나 + API별 활용신청) ---
    ("data.go.kr", "기상청 ASOS 일자료", "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList",
     dict(serviceKey=K, dataType="JSON", dataCd="ASOS", dateCd="DAY", startDt="20250101", endDt="20250102", stnIds="108", numOfRows=2, pageNo=1)),
    ("data.go.kr", "기상청 ASOS 시간자료", "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList",
     dict(serviceKey=K, dataType="JSON", dataCd="ASOS", dateCd="HR", startDt="20250101", startHh="00", endDt="20250101", endHh="02",
          stnIds="108", numOfRows=2, pageNo=1)),
    ("data.go.kr", "축평원 축산물유통정보(LPStock)", "http://data.ekape.or.kr/openapi-data/service/user/grade/LPStock",
     dict(serviceKey=K, issueNo="", numOfRows=1, pageNo=1)),
    ("data.go.kr", "축평원 등급판정(소 경락)", "http://data.ekape.or.kr/openapi-data/service/user/grade/auct/cattle",
     dict(serviceKey=K, startYmd="20250901", endYmd="20250901", numOfRows=1, pageNo=1)),
    ("data.go.kr", "수협 산지조합 위판장 정보", "https://apis.data.go.kr/1192000/select0020List/getselect0020List",
     dict(serviceKey=K, type="json", pageNo=1, numOfRows=1)),
    ("data.go.kr", "aT 일별 도·소매 가격 (배추)", "https://apis.data.go.kr/B552845/perDay/price?"
     "cond[exmn_ymd::GTE]=20260922&cond[exmn_ymd::LTE]=20260922&cond[ctgry_cd::EQ]=200&cond[item_cd::EQ]=211",
     dict(serviceKey=K, returnType="json", pageNo=1, numOfRows=1)),
    ("data.go.kr", "aT 공영도매시장 정산 (가락)", "https://apis.data.go.kr/B552845/katSale/trades?"
     "cond[trd_clcln_ymd::EQ]=2026-09-22&cond[whsl_mrkt_cd::EQ]=110001",
     dict(serviceKey=K, returnType="json", pageNo=1, numOfRows=1)),
    ("data.go.kr", "관세청 품목별 국가별 수출입", "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList",
     dict(serviceKey=K, strtYymm="202501", endYymm="202501", hsSgn="0303")),
    ("data.go.kr", "관세청 품목별 수출입", "https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList",
     dict(serviceKey=K, strtYymm="202501", endYymm="202501", hsSgn="0303")),
    # --- 사이트별 키 ---
    ("KAMIS", "일별 부류별 도소매가격", "http://www.kamis.or.kr/service/price/xml.do",
     dict(action="dailyPriceByCategoryList", p_product_cls_code="02", p_country_code="1101", p_regday="2025-09-01",
          p_convert_kg_yn="N", p_item_category_code="200", p_cert_key=ENV.get("KAMIS_CERT_KEY", ""),
          p_cert_id=ENV.get("KAMIS_CERT_ID", ""), p_returntype="json")),
    ("ECOS", "원/달러 환율(일)",
     f"https://ecos.bok.or.kr/api/StatisticSearch/{ENV.get('ECOS_API_KEY', '')}/json/kr/1/3/731Y001/D/20250102/20250106/0000001", None),
    ("KOSIS", "통계목록", "https://kosis.kr/openapi/statisticsList.do",
     dict(method="getList", apiKey=ENV.get("KOSIS_API_KEY", ""), vwCd="MT_ZTITLE", parentListId="", format="json", jsonVD="Y")),
    ("MAFRA", "채소류 생산실적",
     f"http://211.237.50.150:7080/openapi/{ENV.get('MAFRA_API_KEY', '')}/json/{ENV.get('MAFRA_GRID_VEG_PRODUCTION', '')}/1/2", None),
    ("MAFRA", "농수축산물 도매가격 (EXAMIN_DE 필수)",
     f"http://211.237.50.150:7080/openapi/{ENV.get('MAFRA_API_KEY', '')}/json/{ENV.get('MAFRA_GRID_WHOLESALE_PRICE', '')}/1/2",
     dict(EXAMIN_DE="20250901")),
    ("MAFRA", "가축질병 발생정보",
     f"http://211.237.50.150:7080/openapi/{ENV.get('MAFRA_API_KEY', '')}/json/{ENV.get('MAFRA_GRID_LIVESTOCK_DISEASE', '')}/1/2", None),
    ("NIFS", "실시간 관측소 코드(risaCode)", "https://www.nifs.go.kr/OpenAPI_json",
     dict(id="risaCode", key=ENV.get("NIFS_RISA_API_KEY", ""))),
    ("NIFS", "적조 발생(redtideList)", "https://www.nifs.go.kr/OpenAPI_json",
     dict(id="redtideList", key=ENV.get("NIFS_REDTIDE_API_KEY", ""), sdate="20250801", edate="20250831")),
    ("NIFS", "어장환경 해양(femoSeaList)", "https://www.nifs.go.kr/OpenAPI_json",
     dict(id="femoSeaList", key=ENV.get("NIFS_FEMO_API_KEY", ""), sdate="20250801", edate="20250831")),
    ("NIFS", "해파리(jellyList)", "https://www.nifs.go.kr/OpenAPI_json",
     dict(id="jellyList", key=ENV.get("NIFS_JELLY_API_KEY", ""), sdate="20250801", edate="20250831")),
    ("NIFS", "연안정지 관측소 코드(cooCode)", "https://www.nifs.go.kr/OpenAPI_json",
     dict(id="cooCode", key=ENV.get("NIFS_COO_API_KEY", ""), gru_nam="S")),
    ("OPINET", "전국 평균 유가", "https://www.opinet.co.kr/api/avgAllPrice.do",
     dict(out="json", code=ENV.get("OPINET_API_KEY", ""))),
]


def main():
    only = sys.argv[1:]
    for site, name, url, params in TESTS:
        if only and site not in only:
            continue
        status, body = call(url, params)
        flat = re.sub(r"\s+", " ", mask(body))
        if status == 200 and not re.search(FAIL, flat) and len(flat) > 40:
            verdict = "OK"
        else:
            verdict = "FAIL"
        print(f"[{verdict}] {site} | {name} | HTTP {status}\n      {flat[:300]}\n")


if __name__ == "__main__":
    main()
