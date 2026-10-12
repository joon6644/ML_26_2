"""분야별 모델 입력 통합 데이터 → data/processed/{agri,livestock,fishery}/model_input.parquet (+ .json 피처 목록)

    python scripts/build_model_input.py

모델이 실제로 쓰는 행만 한 파일로 모은다 (확정 구성 v2):
  - 학습 (split = "학습", 2016-01-01 ~ 2024-12-01): 정제 데이터 data/processed_R/{분야}/model_frame_test.parquet
      하루짜리 튐(직전 조사일 대비 1.5배 이상 튀었다가 다음 조사일에 원래 수준으로 돌아온 날)을 직전 조사일 가격으로 대치한 데이터
  - 테스트 (split = "테스트", 2025-01-01 ~): 원래 데이터 data/processed/{분야}/model_frame_test.parquet
      오늘 튄 값이 하루짜리인지는 다음 날에야 알 수 있으므로 예측·평가는 원래 데이터로
  - 2024-12-02 ~ 12-31은 비움: 학습 행의 타깃(향후 28일)이 테스트 기간으로 넘어가지 않게
  - 타깃 y가 없는 행은 뺀다
열: 식별(날짜·품목·품종·등급·소매/중도매·지역·is_nat), split, y, 피처 전부, base(최근 7일 평균), _scope__*(품목 대응 피처의 대상 여부)
정제 데이터는 eda/68_outlier_removal.py 로 만든다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P, PR = ROOT / "data" / "processed", ROOT / "data" / "processed_R"
TRAIN = ("2016-01-01", "2024-12-01")
TEST_START = "2025-01-01"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for d, nm in NAMES.items():
        r = pd.read_parquet(PR / d / "model_frame_test.parquet")
        o = pd.read_parquet(P / d / "model_frame_test.parquet")
        r = r[(r.date >= TRAIN[0]) & (r.date <= TRAIN[1]) & r.y.notna()].assign(split="학습")
        o = o[(o.date >= TEST_START) & o.y.notna()].assign(split="테스트")
        df = pd.concat([r, o[r.columns]], ignore_index=True)
        cols = ["date", "split"] + [c for c in df.columns if c not in ("date", "split")]
        df[cols].to_parquet(P / d / "model_input.parquet", index=False)
        meta = json.loads((P / d / "model_frame_test.json").read_text(encoding="utf-8"))
        meta.update({"train": list(TRAIN), "test_start": TEST_START})
        (P / d / "model_input.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{nm}: data/processed/{d}/model_input.parquet  학습 {len(r):,}행 + 테스트 {len(o):,}행, {df.shape[1]}열")
