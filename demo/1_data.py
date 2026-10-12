"""1단계: 데이터 설명 — 어디서 받았고, 어떻게 가공해, 모델에 무엇이 들어가는가

    python demo/1_data.py                  # 출처 · 가공 흐름 · 최종 입력 데이터 요약 · 샘플 행
    python demo/1_data.py --features 농산물  # 그 분야 모델에 들어가는 피처 목록 (이름 · 의미 · 기대 방향)
"""
import sys

import pandas as pd

from shared import NAMES, P, ROOT, domain, load_frame

RAW = ROOT / "data" / "raw"
SOURCES = [   # (data/raw 폴더, 출처, 내용)
    ("at_price", "KAMIS (aT)", "농·축·수산 소매·중도매 일별 가격 (지역별) — 예측 대상"),
    ("ekape_consumer", "축산물품질평가원", "축산 소비자가격 2016~2017 보완"),
    ("at_trade", "aT 공영도매시장", "가락시장 정산 (경락가·거래 건수·산지)"),
    ("mafra_wholesale", "농식품부", "도매가격"),
    ("asos", "기상청 ASOS", "전 관측소 일별 기온·강수·일조·풍속"),
    ("ekape", "축산물품질평가원", "소 경락가·두수, 돼지 경락가, 재고"),
    ("mafra_disease", "농식품부", "가축 질병 발생"),
    ("customs", "관세청", "품목별 수출입"),
    ("ecos", "한국은행 ECOS", "환율, 소비자물가, 수입물가, 국제 곡물"),
    ("kosis", "통계청 KOSIS", "가축동향, 어업생산, 유가, 작물 생산"),
    ("nifs", "국립수산과학원", "연근해 수온, 양식장 수질"),
    ("calendar", "한국천문연구원", "공휴일·설·추석"),
]


def overview():
    print("\n■ 1. 원본 데이터 출처 (공공 API → data/raw/)\n")
    rows = []
    for folder, src, desc in SOURCES:
        files = [f for f in (RAW / folder).rglob("*") if f.is_file()] if (RAW / folder).exists() else []
        rows.append({"출처": src, "내용": desc, "파일 수": len(files), "용량(MB)": round(sum(f.stat().st_size for f in files) / 1e6)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n■ 2. 가공 흐름\n")
    print("  data/raw (API 원본)\n"
          "   → scripts/build.py          분야별 정제 (원/kg 환산, 공표일 기준 결합 = 미래 정보 차단)\n"
          "   → scripts/build_dataset.py  dataset.parquet  : 가격 + 모든 변수 (원본 값)\n"
          "   → scripts/build_features.py features.parquet : 가격 이력 피처 (변화율·전년 대비·변동성)\n"
          "   → scripts/build_model_input.py model_input.parquet : 모델 입력 통합 데이터 (지역 + 전국 행, 최종 피처, 타깃 y)\n"
          "      학습 행 = 하루짜리 튐을 전날 값으로 대치한 데이터 · 테스트 행 = 원래 데이터")

    print("\n■ 3. 모델 입력 통합 데이터 (data/processed/{분야}/model_input.parquet)\n")
    rows = []
    for d in ["agri", "livestock", "fishery"]:
        df, num = load_frame(d, columns=["date", "split", "item_nm", "sgg_nm", "se_nm", "is_nat", "y"])
        rows.append({"분야": NAMES[d], "행": f"{len(df):,}", "품목": df.item_nm.nunique(), "지역": df[df.is_nat == 0].sgg_nm.nunique(),
                     "소매/중도매": "/".join(sorted(df.se_nm.unique())), "기간": f"{df.date.min():%Y-%m} ~ {df.date.max():%Y-%m}",
                     "학습 행": f"{(df.split == '학습').sum():,}", "테스트 행": f"{(df.split == '테스트').sum():,}", "수치 피처": len(num)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  한 행 = 품목·품종·등급 × 소매/중도매 × 지역(또는 '전국') × 날짜")
    print("  타깃 y = 향후 28일 평균 가격 / 최근 7일 평균 가격 − 1  (4주 뒤 평균이 지금보다 몇 % 변하나)")

    print("\n■ 4. 샘플 — 양파 · 서울 · 소매\n")
    s = pd.read_parquet(P / "agri" / "model_input.parquet", filters=[("item_nm", "==", "양파"), ("sgg_nm", "==", "서울"), ("se_nm", "==", "소매")])
    cols = ["date", "split", "item_nm", "sgg_nm", "base", "y", "r7", "r28", "yoy", "premium", "garak_price_kg", "wx_rain_14d"]
    print(s[cols].dropna().tail(5).round(3).rename(columns={"base": "최근7일평균", "r7": "1주변화", "r28": "4주변화", "yoy": "전년대비",
                                                    "premium": "지역프리미엄", "garak_price_kg": "가락경락가", "wx_rain_14d": "2주강수"}).to_string(index=False))


def features(nm):
    f = pd.read_csv(ROOT / "data" / "reference" / "feature_selection.csv")
    f = f[f.분야 == nm]
    print(f"\n■ {nm} 모델 피처 ({len(f)}개, data/reference/feature_selection.csv)\n")
    pd.set_option("display.max_colwidth", 60)
    print(f[["그룹", "feature", "한글명", "기대 방향"]].to_string(index=False))
    extra = {"농산물": "도매-소매 괴리 이상치, 최근 2주 강수 누적, 최근 2주 일조·기온의 평년 대비 (+4)",
             "수산물": "물가·환율·유가 지수의 수준값 → 4주·13주 변화율로 교체", "축산물": "없음"}
    print(f"\n  + 실험으로 채택한 파생 변수: {extra[nm]}  (근거: eda/derived_log.md)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--features":
        features(NAMES[domain(a[1])] if len(a) > 1 else "농산물")
    else:
        overview()
