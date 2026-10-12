"""발표 시연용 공통 도구: 최종 데이터 읽기, 전처리(학습 때와 동일), 저장 모델 불러오기, 지표."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
MODELS = ROOT / "models"
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
BY_NAME = {v: k for k, v in NAMES.items()}
TRAIN = ("2016-01-01", "2024-12-01")   # 학습 구간 (끝을 12-01로 둬 학습 타깃이 테스트 기간으로 넘어가지 않게)
TEST_START = "2025-01-01"
BAND = 0.01                           # 방향 기준 ±1% (이내면 보합)
SIZE_BINS = [(0.01, 0.02, "1~2%"), (0.02, 0.05, "2~5%"), (0.05, 0.10, "5~10%"), (0.10, 9.9, "10% 이상")]   # 예측 크기 구간 (표시용)
sys.stdout.reconfigure(encoding="utf-8")



def domain(arg):
    """'agri' 또는 '농산물' 어느 쪽으로 받아도 영문 코드로."""
    return BY_NAME.get(arg, arg)


def load_frame(d, columns=None):
    """분야별 모델 입력 통합 데이터 (data/processed/{d}/model_input.parquet: 학습 = 정제 데이터, 테스트 = 원래 데이터) + 피처 목록."""
    path = P / d / "model_input.parquet"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    df = pd.read_parquet(path, columns=columns)
    return df, meta["num"]


def label(v):
    return np.where(v > BAND, 1, np.where(v < -BAND, -1, 0))


OUT_OF_SCOPE = -1e9   # 대상 밖 품목(예: 양파 행의 가축 질병)의 값 — 어떤 실제 값보다 작아 트리가 "해당 없음"으로 따로 나눔


def features(df, num, cats, categories=None, seed=42):
    """LightGBM 입력: 원래 값 그대로 (트리는 값의 순서만 보므로 스케일 변환 없음), 결측은 채우지 않음,
    대상 밖 품목 = OUT_OF_SCOPE, 범주형. 피처 순서는 실험 때처럼 시드로 섞음 (colsample 재현용)."""
    num = sorted(num)
    num = [num[i] for i in np.random.default_rng(seed).permutation(len(num))]
    X = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    for c in num:
        s = f"_scope__{c}"
        if s in df:
            X.loc[~df[s].astype(bool).values, c] = OUT_OF_SCOPE
    for c in cats:
        cat = categories[c] if categories else sorted(df[c].astype(str).unique())
        X[c] = df[c].astype(str).astype(pd.CategoricalDtype(cat))
    return X


def transform(df, meta):
    """예측 때 입력: 학습 때와 같은 features() (같은 피처 순서·범주)."""
    X = features(df, meta["num"], meta["cats"], meta["categories"], meta["seed"])
    return X[meta["columns"]]


def load_models(d):
    import joblib
    path = MODELS / f"{d}.joblib"
    if not path.exists():
        sys.exit(f"저장된 모델이 없습니다: {path.relative_to(ROOT)} → 먼저 python demo/0_train_models.py {d}")
    return joblib.load(path)


def predict(bundle, df):
    """LightGBM 예측 (향후 4주 평균 / 최근 7일 평균 − 1)."""
    return bundle["model"].predict(transform(df, bundle["meta"]))


def metrics(y, p):
    mv = np.abs(y) > BAND
    return {"RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MAE": float(np.mean(np.abs(p - y))),
            "R²": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "방향 정확도": float(np.mean(np.sign(p[mv]) == np.sign(y[mv]))) if p.any() else np.nan}


def size_bin(p):
    """예측 크기 구간 이름 (보합 = ±1% 이내)."""
    a = np.abs(p)
    out = np.full(len(a), "보합 (±1% 이내)", dtype=object)
    for lo, hi, name in SIZE_BINS:
        out[(a > lo) & (a <= hi)] = name
    return out


def load_prob(d):
    """예측 크기 → '실제로 그 방향으로 1% 넘게 움직일 확률' 보정기 (eda/73_direction_prob.py, 학습에 안 쓴 예측 중 정답이 나온 것 전부로 맞춤)."""
    import joblib
    path = MODELS / f"prob_{d}.joblib"
    if not path.exists():
        sys.exit(f"확률 보정기가 없습니다: {path.relative_to(ROOT)} → 먼저 python eda/73_direction_prob.py")
    return joblib.load(path)


def direction_prob(cal, p):
    """표시용 확률 = a + b·ln|ŷ| (선형회귀), 1% 단위, 분야별 상한(상위 1% 예측의 실제 적중률)까지. 보합(±1% 이내) 예측은 NaN."""
    p = np.asarray(p, dtype=float)
    raw = np.clip(cal["a"] + cal["b"] * np.log(np.maximum(np.abs(p), 1e-6)), 0, 1)
    prob = np.minimum(np.round(raw / cal["step"]) * cal["step"], cal["cap"])
    return np.where(np.abs(p) > BAND, prob, np.nan)


def width(s):
    """터미널에서 차지하는 칸 수 (한글 등 전각 문자는 2칸)."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w, right=False):
    """화면 폭 기준으로 w칸에 맞춤 (pandas·f-string은 한글을 1칸으로 세서 열이 어긋남)."""
    s = str(s)
    fill = " " * max(w - width(s), 0)
    return fill + s if right else s + fill


def table(rows, right=()):
    """dict 목록 → 열 정렬된 표 문자열. right에 든 열은 오른쪽 정렬(숫자)."""
    cols = list(rows[0])
    w = {c: max(width(c), *(width(r[c]) for r in rows)) for c in cols}
    line = lambda r: "  ".join(pad(r[c], w[c], c in right) for c in cols)
    return "\n".join([line({c: c for c in cols}), "  ".join("-" * w[c] for c in cols)] + [line(r) for r in rows])


def show(df, index=False):
    """DataFrame을 한글 폭에 맞춰 출력 (숫자 열은 오른쪽 정렬)."""
    import re
    df = df.reset_index() if index else df
    looks_num = lambda s: s.dropna().astype(str).str.fullmatch(r"[-+]?[\d,.]+%?|–").all()   # "6,365,641"·"72%" 같은 문자열 숫자도 오른쪽
    right = tuple(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or looks_num(df[c]))
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):                           # 소수는 열마다 같은 자릿수 (100 이상은 정수 + 쉼표)
            fmt = "{:,.0f}" if df[c].abs().max() >= 100 else "{:.3f}"
            df[c] = df[c].map(lambda v: "–" if pd.isna(v) else fmt.format(v))
    print(table(df.astype(object).where(df.notna(), "–").to_dict("records"), right=right))


def prob_text(prob, p):
    """예측 카드 문구: 50% 이하면 주의 문구."""
    if np.isnan(prob):
        return "큰 변화 없을 것으로 예상"
    if prob <= 0.5:
        return "신뢰성이 낮으니 판단에 주의하세요"
    return f"4주 뒤 평균이 지금보다 1% 넘게 {'오를' if p > 0 else '내릴'} 확률 약 {prob:.0%}"
