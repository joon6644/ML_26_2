"""서비스 모델(LightGBM) 설정 — 학습·예측 코드가 읽는 유일한 설정 파일.

결정 근거는 eda/derived_log.md ('★ 확정 구성 v3', 'LightGBM 입력에서 MinMax 제거'). 실험 스크립트(eda/*.py)에 남아 있는
다른 설정값(앙상블, Huber, 결측 채우기, MinMax 등)은 과거 실험의 기록이며 현재 설정이 아니다.
"""

DOMAINS = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}

# 범주형 피처 (축산은 소매만 있고 식재료 계열이 하나라 품목·지역만)
CATS = {
    "agri": ["item_cd", "food_group", "se_nm", "sgg_nm"],
    "livestock": ["item_cd", "sgg_nm"],
    "fishery": ["item_cd", "food_group", "se_nm", "sgg_nm"],
}

# LightGBM 하이퍼파라미터 — 기본값 유지 (튜닝 개선이 시드 잡음 2배 미만). random_state는 학습 때 42로 덮어씀
LGB_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8,
              "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}
SEED = 42

# 단조 제약: 상대값 피처의 영향 방향을 가설대로 고정 (+1 = 클수록 4주 변화율이 커짐, −1 = 되돌림).
# 농산은 되돌림이 조건부(공급 충격 중엔 오른 가격이 더 오름)라 제약 없음. 원/kg·두수 같은 수준값 피처에는 걸지 않음.
MONO = {
    "agri": {},
    "livestock": {"dev_last": 1, "r91": -1, "yoy": -1, "r28": -1, "disease_item_cases_28d": 1, "pig_auction_price_kg": 1,
                  "usd_krw": 1, "premium": -1, "nat_r91": -1, "nat_yoy": -1, "nat_r28": -1, "nat_ly_change": 1},
    "fishery": {"dev_last": 1, "yoy": -1, "r91": -1, "ly_change": 1, "farm_cod_s": 1, "farm_do_s_b": -1, "farm_din_dip_s": -1,
                "premium": -1, "nat_yoy": -1, "nat_ly_change": 1, "nat_dev_last": 1},
}

TARGET_CLIP = (0.01, 0.99)   # 학습 타깃(4주 변화율) 상하위 1%를 경계값으로 — 학습 때만, 평가·예측은 원래 값


def lgb_params(domain, columns):
    """분야·피처 순서에 맞는 LightGBM 인자 (단조 제약 포함)."""
    p = {**LGB_PARAMS, "random_state": SEED}
    mono = MONO[domain]
    if mono:
        p.update(monotone_constraints=[mono.get(c, 0) for c in columns], monotone_constraints_method="advanced")
    return p
