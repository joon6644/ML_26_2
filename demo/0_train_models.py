"""발표 전 준비 (시연하지 않음): 최종 구성으로 모델 학습 → models/{분야}.joblib 저장

    python demo/0_train_models.py              # 세 분야 전부 (약 2분) — 발표 전에 1회
    python demo/0_train_models.py livestock    # 한 분야만

최종 구성 (eda/derived_log.md '★ 확정 구성 v3', 2026-10-10):
  - 모델: LightGBM 하나, 시드 42, 기본 하이퍼파라미터
      (LGB+XGB+Cat 앙상블은 이득이 제한적 — 농산 R² +0.01~0.02, 축·수는 잡음 수준 — 이라 시스템 복잡도를 줄이려고 단일 모델로)
  - 데이터 정제 (세 분야 공통): 하루짜리 튐(직전 조사일 대비 1.5배 이상 튀었다가 다음 조사일에 원래 수준으로 돌아온 날)은
    직전 조사일 가격으로 대치 → data/processed_R/ (eda/68_outlier_removal.py 로 생성)
  - 학습 처리 (세 분야 공통): 학습 타깃 상하위 1% 자르기
  - 결측 처리: 없음 (NaN 그대로, 트리가 비어 있는 경우를 따로 갈라 학습) — eda/71_no_impute.py
  - 단조 제약: 축·수산만, 상대값 피처에 가설 방향 고정
입력 = data/processed/{분야}/model_input.parquet (scripts/build_model_input.py)
  학습 행 = 정제 데이터 2016-01 ~ 2024-12-01 / 테스트 행 = 원래 데이터 2025-01~ (오늘 튄 값이 하루짜리인지는 다음 날에야 알 수 있으므로)
저장 후 2025~ 테스트 예측을 eda/tables/final_test_preds_{분야}.npz 로 남긴다 (성능표는 eda/70_final_v2.py).
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from shared import MODELS, NAMES, ROOT, TRAIN, domain, load_frame, predict

ns = {"__file__": str(ROOT / "eda" / "60_mono_scope.py")}
exec(open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0], ns)   # 실험 때 쓴 학습 도구 그대로
dv, LEVEL = ns["dv"], ns["LEVEL"]
import lightgbm as lgb   # noqa: E402


def prep_nofill(df, num, tr_m):
    """결측을 채우지 않는 전처리: 무한대만 NaN으로, MinMax는 학습 구간 기준 (NaN은 건너뛰고 그대로 둠)."""
    X = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X).astype("float32"), columns=num, index=df.index)


dv.cs.rm.prep = prep_nofill   # dv.matrix 가 쓰는 전처리 (원래는 학습 평균으로 결측 채움)


def train(d):
    t0 = time.time()
    nm = NAMES[d]
    df, num = load_frame(d)                                               # 통합 데이터: split = 학습(정제 데이터) / 테스트(원래 데이터)
    R, E = df[df.split == "학습"], df[df.split == "테스트"]
    df = pd.concat([R, E], ignore_index=True)
    df.attrs["scope"] = {c[8:]: df[c].values.astype(bool) for c in df.columns if c.startswith("_scope__")}
    cats = dv.base_sets(nm)[1]
    tr = np.r_[np.ones(len(R), bool), np.zeros(len(E), bool)]
    yv = df.y.values.astype(float)
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], (0.01, 0.99)))         # 학습 타깃 상하위 1% 자르기
    mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
    print(f"[{nm}] 학습 행 {tr.sum():,} (정제 데이터) / 피처 {len(num) + len(cats)}개 / 타깃 1~99% 자르기, 단조 제약 {len(mono)}개", flush=True)

    X = dv.matrix(df, num, cats, tr, 42)   # 피처 순서 섞기·전처리까지 실험 때와 동일 (MinMax는 학습 행 기준)
    pl = {**dv.LGB_BASE, "random_state": 42}
    if mono:
        pl.update(monotone_constraints=[mono.get(c, 0) for c in X.columns], monotone_constraints_method="advanced")
    t = time.time()
    model = lgb.LGBMRegressor(**pl).fit(X[tr], yt, categorical_feature=cats)
    print(f"  LightGBM {time.time() - t:.0f}초", flush=True)

    # 예측 때 같은 전처리를 하기 위한 값 (학습 구간 기준)
    order = [c for c in X.columns if c not in cats]
    Xn = df[order].astype("float32").replace([np.inf, -np.inf], np.nan)
    meta = {"num_order": order, "cats": cats, "columns": list(X.columns),
            "scaler": MinMaxScaler().fit(Xn[tr]),
            "scope": list(df.attrs["scope"]), "categories": {c: list(X[c].cat.categories) for c in cats}}
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
