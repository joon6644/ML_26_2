"""발표 전 준비 (시연하지 않음): 서비스 모델 학습 → models/{분야}.joblib 저장

    python demo/0_train_models.py              # 세 분야 전부 (약 2분)
    python demo/0_train_models.py livestock    # 한 분야만

설정은 scripts/model_config.py 하나뿐이다 (결정 근거: eda/derived_log.md '★ 확정 구성 v3' 이후 기록).
  - 모델: LightGBM 하나, 시드 42, 기본 하이퍼파라미터
  - 데이터 정제: 하루짜리 튐(직전 조사일 대비 1.5배 이상 튀었다가 다음 조사일에 원래 수준으로 돌아온 날)은 직전 조사일 가격으로
    → 학습 행만 (data/processed_R/, eda/68_outlier_removal.py 로 생성)
  - 학습 타깃 상하위 1% 자르기, 결측 채우지 않음, 스케일 변환 없음, 단조 제약은 축·수산만
입력 = data/processed/{분야}/model_input.parquet (scripts/build_model_input.py)
  학습 행 = 정제 데이터 2016-01 ~ 2024-12-01 / 테스트 행 = 원래 데이터 2025-01~ (오늘 튄 값이 하루짜리인지는 다음 날에야 알 수 있으므로)
저장 후 2025~ 테스트 예측을 eda/tables/final_test_preds_{분야}.npz 로 남긴다 (성능표는 eda/72_final_lgb.py).
"""
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from shared import MODELS, NAMES, ROOT, TRAIN, domain, features, load_frame, predict

sys.path.insert(0, str(ROOT / "scripts"))
from model_config import CATS, MONO, SEED, TARGET_CLIP, lgb_params   # noqa: E402


def train(d):
    t0 = time.time()
    nm = NAMES[d]
    df, num = load_frame(d)                                               # 통합 데이터: split = 학습(정제 데이터) / 테스트(원래 데이터)
    R, E = df[df.split == "학습"], df[df.split == "테스트"]
    df = pd.concat([R, E], ignore_index=True)
    cats = CATS[d]
    tr = np.r_[np.ones(len(R), bool), np.zeros(len(E), bool)]
    yv = df.y.values.astype(float)
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], TARGET_CLIP))            # 학습 타깃 상하위 1% 자르기
    print(f"[{nm}] 학습 행 {tr.sum():,} (정제 데이터) / 피처 {len(num) + len(cats)}개 / 단조 제약 {len(MONO[d])}개", flush=True)

    X = features(df, num, cats, seed=SEED)                                # 원래 값 그대로, 대상 밖 품목 = OUT_OF_SCOPE, 범주형
    t = time.time()
    model = lgb.LGBMRegressor(**lgb_params(d, X.columns)).fit(X[tr], yt, categorical_feature=cats)
    print(f"  LightGBM {time.time() - t:.0f}초", flush=True)

    # 예측 때 같은 입력을 만들기 위한 값 (피처 목록·순서 시드·범주)
    meta = {"num": num, "cats": cats, "seed": SEED, "columns": list(X.columns),
            "categories": {c: list(X[c].cat.categories) for c in cats}}
    MODELS.mkdir(exist_ok=True)
    bundle = {"model": model, "meta": meta, "domain": d, "train": TRAIN}
    joblib.dump(bundle, MODELS / f"{d}.joblib")

    # 2025~ 테스트 예측 저장 + 확인: 저장한 모델·전처리로 다시 예측한 값 = 학습 직후 예측
    direct = model.predict(X[~tr])
    gap = float(np.max(np.abs(predict(bundle, E) - direct)))
    np.savez(ROOT / "eda" / "tables" / f"final_test_preds_{d}.npz", lgb=direct)
    print(f"  저장 모델 재예측 vs 학습 직후 예측 최대 차이 {gap:.2e} ({'일치' if gap < 1e-4 else '불일치 — 확인 필요'})")
    print(f"[{nm}] 저장 → models/{d}.joblib ({time.time() - t0:.0f}초)\n", flush=True)


if __name__ == "__main__":
    for d in [domain(a) for a in sys.argv[1:]] or ["livestock", "fishery", "agri"]:
        train(d)
