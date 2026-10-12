# [파이프라인 모듈] 학습 데이터(model_frame·processed_R)를 만들 때 실행된다 — 지우지 말 것. 이 파일에 남은 학습 설정(전처리·모델·하이퍼파라미터)은 과거 실험값이며, 현재 서비스 설정은 scripts/model_config.py (eda/README.md)
# 품종명 변경 연결(data/reference/variety_rename.csv, scripts/build.py link_renamed) 뒤 학습 데이터 전체 재구성 (2026-10-11)
#   python eda/74_rebuild_after_rename.py frames_v1   # data/processed/{분야}/model_frame_v1 (검증 분할: 정규화 ~2022-12-01)
#   python eda/74_rebuild_after_rename.py frames_test # data/processed/{분야}/model_frame_test (테스트 분할: 정규화 ~2024-12-01)
#   python eda/74_rebuild_after_rename.py clean       # data/processed_R (하루짜리 튐 → 전날 값) model_frame_test + model_frame_v1
#   python eda/74_rebuild_after_rename.py val         # 검증 예측 (LightGBM, 원래 값 입력, ~2022-12-01 정제 데이터 학습 → 2023~24 원래 데이터) → eda/tables/final_val_preds_{분야}.npz
# 앞뒤 단계: scripts/build_dataset.py → build_features.py → (이 스크립트) → scripts/build_model_input.py → demo/0_train_models.py → eda/72 → eda/73
import importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb   # noqa: E402
P0, PR, T = ROOT / "data" / "processed", ROOT / "data" / "processed_R", ROOT / "eda" / "tables"
NAMES = dv.cs.rm.NAMES
VAL = ("2016-01-01", "2022-12-01", "2023-01-01", "2024-12-31")
TEST = ("2016-01-01", "2024-12-01", "2025-01-01", "2026-12-31")


def save_frame(df, num, cats, path):
    keep = sorted(set(dv.cs.KEYS + ["sgg_nm", "date", "y", "is_nat", "food_group", "dev_last", "base"] + num + cats))
    out = df[[c for c in keep if c in df.columns]].copy()
    for c, ok in df.attrs.get("scope", {}).items():
        out[f"_scope__{c}"] = ok
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.attrs = {}
    out.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"num": num, "cats": cats}, ensure_ascii=False), encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}  {len(out):,}행", flush=True)


def frames(root, split, name):
    dv.cs.rm.P = dv.cs.P = dv.cs.fs.P = root
    dv.cs.rm.SPLIT = dv.cs.SPLIT = dv.cs.fs.SPLIT = split
    for d in NAMES:
        df, num, cats = dv.final_frame(d)
        save_frame(df, num, cats, root / d / f"{name}.parquet")


def prep_nofill(df, num, tr_m):
    X = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X).astype("float32"), columns=num, index=df.index)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "frames_v1":
        frames(P0, VAL, "model_frame_v1")
    elif step == "frames_test":
        frames(P0, TEST, "model_frame_test")
    elif step == "clean":
        for old in ("processed_A", "processed_B", "processed_R"):
            shutil.rmtree(ROOT / "data" / old, ignore_errors=True)
        spec = importlib.util.spec_from_file_location("o68", ROOT / "eda" / "68_outlier_removal.py")
        o68 = importlib.util.module_from_spec(spec); spec.loader.exec_module(o68)
        o68.build_variant("R")                                            # 정제 가격 → 데이터셋 → 피처 → model_frame_test
        bd, bfe = o68.bd, o68.bfe
        bd.P = bfe.P = P0                                                 # 원래 경로로 되돌림
        frames(PR, VAL, "model_frame_v1")                                 # 검증용 정제 학습 표
    elif step == "val":                                                  # 서비스 모델과 같은 입력 (원래 값, MinMax 없음 — 2026-10-12)
        sys.path.insert(0, str(ROOT / "demo"))
        from shared import features
        from model_config import CATS, SEED, TARGET_CLIP, lgb_params   # 서비스 모델과 같은 설정 파일
        for d in NAMES:
            O = pd.read_parquet(P0 / d / "model_frame_v1.parquet")
            E = O[(O.date >= VAL[2]) & (O.date <= VAL[3]) & O.y.notna()]
            R = pd.read_parquet(PR / d / "model_frame_v1.parquet")
            R = R[(R.date >= VAL[0]) & (R.date <= VAL[1]) & R.y.notna()]
            num = json.loads((P0 / d / "model_frame_v1.json").read_text(encoding="utf-8"))["num"]
            cats = CATS[d]
            df = pd.concat([R, E[R.columns]], ignore_index=True)
            tr = np.r_[np.ones(len(R), bool), np.zeros(len(E), bool)]
            yv = df.y.values.astype(float)
            yt = np.clip(yv[tr], *np.nanquantile(yv[tr], TARGET_CLIP))
            X = features(df, num, cats, seed=SEED)
            p = lgb.LGBMRegressor(**lgb_params(d, X.columns)).fit(X[tr], yt, categorical_feature=cats).predict(X[~tr])
            np.savez(T / f"final_val_preds_{d}.npz", lgb=p)
            print(f"  {NAMES[d]} 검증 예측 {len(p):,}행", flush=True)
