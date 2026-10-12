# eda/ 지도

이 폴더에는 세 종류의 파일이 섞여 있다. 각 `.py` 첫 줄의 `[파이프라인 모듈]` / `[현재 결과]` / `[기록용 실험]` 표시가 이 구분이다.

**현재 서비스 설정은 `scripts/model_config.py` 하나뿐이다.** 실험 스크립트 안에 보이는 다른 설정(MinMax, 결측 평균 채우기, Huber, XGBoost·CatBoost 앙상블, 신뢰도 등급, 85% 상한 등)은 모두 과거 실험값이다.

## 1. 파이프라인 모듈 — 지우지 말 것

학습 데이터(`data/processed/*/model_frame_*`, `data/processed_R/`)를 만들 때 실제로 실행된다. 서비스 모델 학습(`demo/0_train_models.py`)은 이 모듈들을 부르지 않고 `model_input.parquet`과 `scripts/model_config.py`만 읽는다.

| 파일 | 역할 |
| --- | --- |
| `13_regional_model.py` | 지역+전국 행 구성, 기본 전처리 함수 (`rm`) |
| `21_final_selection.py` | 피처 목록·대상 품목 범위 (`fs`) |
| `22_combined_selection.py` | 통합 학습 표 만들기 (`cs`) — 13·21을 불러 씀 |
| `24_derived.py` | 파생 변수를 붙인 최종 학습 표 `final_frame()` (`dv`) — 22를 불러 씀 |
| `60_mono_scope.py` | 위 모듈을 한 번에 불러오는 진입점 (`dv`, `LEVEL`) |
| `68_outlier_removal.py` | 하루짜리 튐을 전날 값으로 바꾼 정제 데이터 `data/processed_R/` 만들기 |
| `74_rebuild_after_rename.py` | 원본이 바뀌었을 때 학습 표·정제 데이터·검증 예측을 다시 만드는 순서 |

## 2. 현재 결과 — 지금 보고서·서비스 숫자를 만드는 스크립트

| 파일 | 만드는 것 |
| --- | --- |
| `72_final_lgb.py` | 서비스 모델 성능표 (`tables/final_eval*.csv`) |
| `73_direction_prob.py` | 카드 확률 계수·상한 (`models/prob_*.joblib`), 그림 `figures/direction_prob_curve.png` |
| `77_business_kpi.py` | 절약·손실 KPI (서비스 모델 기준) |
| `78_report2_rf.py` | 과제2 1차 예측 모델(RandomForest) 검증 결과 |
| `79_val_first_models.py` | 검증 기간 모델 비교 (Ridge / LightGBM 가격 이력만 / + 외부 변수) — 과제3 자료 |
| `80_report2_shopping.py` | 과제2 자취생 장바구니 절약률 |

## 3. 기록용 실험 — 나머지 전부 (01 ~ 71)

번호 순서대로 가설을 세우고 돌린 실험이다. 결과와 판단은 `derived_log.md`에 있고, 각 스크립트는 그 시점의 설정으로 돌았다. **새 작업의 출발점으로 쓰지 말 것** — 현재 설정과 다르다.

## 현재 파이프라인 순서

```
scripts/collect.py → scripts/build.py → scripts/build_dataset.py → scripts/build_features.py
  → eda/74 frames_v1, frames_test, clean (학습 표·정제 데이터) → scripts/build_model_input.py
  → demo/0_train_models.py (서비스 모델) → eda/74 val (검증 예측) → eda/72 (성능표) → eda/73 (확률)
```
