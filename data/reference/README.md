# 기준표 (data/reference/)

사람이 검토하고 고치는 표. git으로 관리한다. `scripts/build.py`가 읽는다.

| 파일 | 내용 | 키 |
|---|---|---|
| `item_map.csv` | aT/KAMIS 품목 101개 → 도메인, 식재료 계열, 영양성분 식품코드, HS코드, KOSIS 생산통계, 검색어 | `ctgry_cd` + `item_cd` |
| `unit_weight.csv` | 개수 단위(포기·개·마리·손·장·속·구) 1개의 무게(g). 가격을 원/kg로 바꿀 때 사용 | `item_nm` + `unit` (+ `vrty_contains`) |
| `asos_station_sido.csv` | ASOS 관측소 97곳 → 시도 (고랭지 등 비고) | `stnId` |
| `sido_alias.csv` | KOSIS·주소의 시도 표기 → 짧은 시도명 (강원특별자치도 → 강원 등) | `name` |
| `kma_buoy_station.csv` | 기상청 해양기상부이 중 2016년부터 운영된 17곳 → 지점명·해역. 해역 수온 평균에 이 17곳만 사용 | `stn_id` |
| `ekape_market.csv` | 축평원 경락 도매시장 코드 → 시장명·권역 (축평원 소 도매시장 정보 API). CTot 등 권역 집계 코드는 공식 명칭 미확인 | `market_cd` |

## item_map.csv 컬럼

| 컬럼 | 설명 |
|---|---|
| domain | agri / livestock / fishery |
| food_group | 식재료 계열 (엽채류, 근채류, 과채류, 양념채소, 버섯류, 곡류, 두류, 서류, 견과·종실류, 과일, 과일(수입), 쇠고기, 돼지고기, 가금류, 난류, 유제품, 등푸른생선, 흰살생선, 연체류, 갑각류, 조개류, 해조류, 건어물, 젓갈·조미료). 대체재 후보를 좁히는 1차 필터 |
| nutrition_code / nutrition_name | `nutrition_raw_material.csv`의 식품코드·식품명. 품목의 대표 식품 1개 (생것 우선) |
| hs_prefixes | 관세청 HS코드 앞자리. `;`로 여러 개. 수입 피처를 붙일 때 `hs_code.startswith()`로 매칭 |
| kosis_production | `표이름:항목명` 형식. agri는 production_annual의 table/item_nm, fishery는 production_monthly의 품종별, livestock은 census의 축종 |
| garak_item | 가락시장 정산 품목 `대분류>중분류` (`;` 구분). `agri/trade_by_item_daily.parquet`를 만들 때 사용. 곡류·가공품·수입 견과 등 가락 미거래 품목은 빈칸 |
| search_keywords | 네이버 검색어 트렌드용 (`;` 구분). "무·파·배"처럼 뜻이 여러 개인 단어는 피함 |
| note | 주의사항 |

## unit_weight.csv 규칙

- `vrty_contains`가 비어 있지 않으면 품종명(vrty_nm)에 그 글자가 들어간 행에 먼저 적용하고, 없으면 빈 칸(기본값) 행을 쓴다.
- 값은 KAMIS 「농축산물 유통정보조사요령」「수산물 가격조사 기준」(2023)의 **상품 규격 중간값**. 조사기준에 없는 품목은 `source`에 "추정(확인 필요)"로 표시했다.
- 무게 단위(kg, g, "kg(그물망 3포기)")는 표 없이 그대로 환산한다.

## 추정값

조사기준에 없는 값은 유통 규격을 검색해 채웠다 (source 칸에 근거 표기). 알배기배추 600g, 브로콜리 350g, 삼치 600g, 전어 90g은 온라인·업소용 유통 규격, 무는 KAMIS 조사규격(계절별), 계란은 축산물 등급판정 세부기준 별표23(특란 60~68g).
source가 "추정"인 행은 품종을 모를 때 쓰는 기본값(공식 값들의 평균)과 가지·키위뿐이다. 값을 고치면 `python scripts/build.py`로 다시 반영된다.
