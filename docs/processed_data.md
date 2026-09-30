# 가공 데이터 (data/processed/)

학습 기간 **2016-01-01 ~ 2026-09-30**. 모든 파일은 parquet이며 `pd.read_parquet(경로)`로 읽는다.
원천별 상세·제공 기간은 [data_inventory.md](data_inventory.md), 품목·단위·관측소 기준표는 [data/reference/](../data/reference/README.md) 참고.

## 만드는 법

```bash
python scripts/collect.py      # 원천 수집 → data/raw/ (끊겨도 이어받음, 한도 초과 시 다음 날 다시 실행)
python scripts/build.py        # 병합 → data/processed/
```

**수동 다운로드 1건**: 해양기상부이 수온은 API가 공공기관 전용이라 [기상자료개방포털 해양기상부이](https://data.kma.go.kr/data/sea/selectBuoyRltmList.do?pgmNo=52)에서 로그인 후 **시간자료, 지점 전체, 요소 수온·풍속**을 1년 단위로 받아 `data/raw/kma_buoy/`에 넣는다 (시간자료는 1년씩만 받아짐).

## 폴더 구조

```
data/processed/
├── common/      날짜 기준 공통 피처 (달력, 기상, 환율, 거시지표)
├── agri/        농산물 (KAMIS 부류 100 식량, 200 채소, 300 특용, 400 과일)
├── livestock/   축산물 (부류 500)
└── fishery/     수산물 (부류 600)
```

각 도메인의 `price.parquet`가 **타깃**, 나머지는 피처 후보다. 피처를 가격에 붙일 때 품목 매핑은 아직 정하지 않았다 (각 파일에 품목명·코드를 그대로 둠).

---

## 타깃: `{agri,livestock,fishery}/price.parquet`

출처: aT 일별 도·소매 가격정보 (data.go.kr). 1행 = 날짜 × 품목 × 품종 × 등급 × 소매/도매 × 지역 × 시장.

| 컬럼 | 설명 |
|---|---|
| date | 조사일 |
| se_cd / se_nm | 구분 (01 소매, 02 중도매, 친환경농산물 등) |
| ctgry_cd / ctgry_nm | 부류 (100~600) |
| item_cd / item_nm | 품목 (예: 211 배추) |
| vrty_cd / vrty_nm | 품종 (예: 여름(고랭지)). **배추·무 등은 계절마다 품종 코드가 바뀜** |
| grd_cd / grd_nm | 등급 (상품, 중품 …) |
| sgg_cd / sgg_nm | 지역 |
| mrkt_cd / mrkt_nm | 시장·매장 |
| unit / unit_sz | 단위와 단위 크기 (예: kg / 20 → 20kg 기준 가격) |
| price | 조사 가격 (원, unit_sz × unit 기준) |
| unit_g / unit_g_source | 단위 1개의 무게(g)와 근거 (weight=무게 단위, table:…=`data/reference/unit_weight.csv`) |
| **price_kg** | **원/kg** = price ÷ (unit_sz × unit_g / 1000). 품목 간 비교·대체재 추천에 사용 |
| price_kg_api | API가 준 kg 환산가. 개수 단위(포기·개)는 환산되지 않은 값이라 참고용 |

## common/

| 파일 | 주기 | 내용 |
|---|---|---|
| calendar.parquet | 일 | holiday_name, is_holiday, is_weekend, **days_to_seollal / days_to_chuseok** (다음 설·추석 당일까지 남은 일수) |
| weather_station_daily.parquet | 일 | ASOS 관측소별: temp_avg/min/max, rain(무강수=0), rain_1h_max, sunshine_hr, solar_rad, humidity, wind_gust, wind_avg, snow_depth, ground_temp |
| weather_national_daily.parquet | 일 | 위 항목의 전 관측소 평균 |
| fx_daily.parquet | 일 | usd_krw (원/달러 매매기준율) |
| recipe_basic.parquet / recipe_ingredient.parquet | - | (후순위) MAFRA 레시피 537개: 요리명, 국가·유형(밥/국 …), 조리시간, 칼로리, 난이도 / 재료 6,104행: recipe_id, 재료명, 분량, 주재료·부재료·양념 구분 |
| nutrition_raw_material.parquet | - | 식약처 원재료 영양성분 3,704개 (식품 대·중·소·세분류, 영양성분 약 25종, 폐기율). 품목 연결은 `data/reference/item_map.csv`의 nutrition_code |
| nutrition_food.parquet | - | 음식(요리) 영양성분 19,495개 (후순위) |
| macro_monthly.parquet | 월 | long 형식: table(cpi 소비자물가 식료품 품목, import_price 수입물가 농림수산·식료품, intl_commodity 국제상품가격(원유·옥수수·소맥·대두 등), fuel_price 주유소 평균가), code, name, unit, month, value |

## agri/

| 파일 | 주기 | 내용 |
|---|---|---|
| price.parquet | 일 | 타깃 (위 참고) |
| wholesale_mafra.parquet | 일 | MAFRA 전국 도매시장 가격: 품목·품종·등급·단위·지역·시장·price |
| trade_garak_daily.parquet | 일 | **가락시장 거래량** (2018-01~): 일 × 상품 대/중/소분류 × 단위별 qty(총물량), amount(총금액), n_trades, avg_price |
| **trade_by_item_daily.parquet** | 일 | 위를 **aT 품목 코드로 합산** (62개 품목): qty_kg(kg 단위 거래 물량), amount, n_trades, avg_price_kg. 가격 테이블과 `ctgry_cd`+`item_cd`+`date`로 바로 조인 |
| production_annual.parquet | 연 | KOSIS 채소(엽채·근채·조미)·식량작물·과실 면적·생산량 (long) |
| import_monthly.parquet | 월 | 관세청 HS 07·08·10·12 등 10자리 품목별 imp_kg, imp_usd, exp_kg, exp_usd |

## livestock/

| 파일 | 주기 | 내용 |
|---|---|---|
| price.parquet | 일 | 타깃 |
| auction_monthly.parquet | 월 | 축평원 소·돼지 등급별 × 도매시장별 경락가(price, 원/kg)·두수(head_count). market_cd/market_nm/region: 개별 도매시장명은 `data/reference/ekape_market.csv`, CTot=전체 합계 |
| pig_rep_price_daily.parquet | 일 | 돈육 대표가격 (박피/탕박) |
| stock_monthly.parquet | 월 | 축산물 부위별 재고 (2019-05~) |
| disease_events.parquet | 건 | 가축질병 발생 (1995~): date, end_date, disease, species, head_count, region |
| census_quarterly.parquet | 분기 | KOSIS 가축동향: 축종별·시도별 농장수·마리수 |
| import_monthly.parquet | 월 | 관세청 HS 02·04·1601·1602 |

## fishery/

| 파일 | 주기 | 내용 |
|---|---|---|
| price.parquet | 일 | 타깃 |
| wholesale_mafra.parquet | 일 | MAFRA 도매가격 중 수산 |
| **sea_temp_daily.parquet** | 일 | **해역별 수온** (동해·남해·서해): 기상청 해양기상부이 중 2016년부터 운영된 17곳의 일평균. water_temp, n_buoys |
| buoy_temp_daily.parquet | 일 | 부이별 일평균: stn_id, stn_nm, sea, water_temp, water_temp_max, wind, n_hours(관측 시간 수), core(해역 평균에 쓰는 17곳 여부) |
| coast_temp_daily.parquet | 일 | 연안정지관측 수온. 관측소가 17곳 → 1곳으로 줄어 **피처로 쓰지 않음** (참고용) |
| production_monthly.parquet | 월 | KOSIS 어업생산동향 품종별 생산량·생산금액 |
| redtide_events.parquet | 건 | 적조 발생: 원인생물, 해역, 밀도, 수온 |
| jellyfish_reports.parquet | 주 | 해파리 주간보고 (제목·날짜만) |
| import_monthly.parquet | 월 | 관세청 HS 03·1603~1605 |

---

## 사용 시 주의

- **공개 시차**: 월·분기 데이터(물가, 수입, 생산, 가축동향)는 해당 기간 종료 후 1개월~1분기 뒤에 공개된다. 예측 시점에 알 수 없는 값을 쓰지 않도록 피처를 만들 때 시차만큼 밀어서 붙인다.
- **결측**: 거래량(2018~), 재고(2019-05~), 신규 품목(알배기배추·브로콜리 2023~, 수산 신규 품목 2022.7~)은 앞 기간이 비어 있다. null로 두고 학습한다.
- **휴장일**: 명절·주말에는 가격 행이 없다.
