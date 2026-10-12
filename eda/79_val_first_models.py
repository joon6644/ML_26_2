# [현재 결과] 지금 보고서·서비스 숫자를 만드는 스크립트 (eda/README.md)
# 검증 기간 비교 (모델·피처 선택은 여기서만): 학습 2016-01~2022-12-01 (정제 데이터) → 검증 2023~24 (원래 데이터), 소매 행
#   선형회귀(Ridge) / LightGBM 가격 이력만 / LightGBM + 외부 변수 → eda/tables/val_first_models.csv
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
from shared import NAMES, features, metrics   # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from model_config import CATS, SEED, TARGET_CLIP, lgb_params   # noqa: E402  서비스 모델과 같은 설정 파일
import lightgbm as lgb   # noqa: E402
P0, PR = ROOT / "data" / "processed", ROOT / "data" / "processed_R"
PRICE = {"dev_last", "r7", "r28", "r91", "yoy", "ly_change", "vol28", "n_obs28", "price_kg", "premium", "grp_r28", "woy", "days_to_seollal", "days_to_chuseok"}
rows = []
for d, nm in NAMES.items():
    O = pd.read_parquet(P0 / d / "model_frame_v1.parquet"); E = O[(O.date >= "2023-01-01") & (O.date <= "2024-12-31") & O.y.notna()]
    R = pd.read_parquet(PR / d / "model_frame_v1.parquet"); R = R[(R.date >= "2016-01-01") & (R.date <= "2022-12-01") & R.y.notna()]
    num = json.loads((P0 / d / "model_frame_v1.json").read_text(encoding="utf-8"))["num"]
    cats = CATS[d]
    df = pd.concat([R, E[R.columns]], ignore_index=True)
    tr = np.r_[np.ones(len(R), bool), np.zeros(len(E), bool)]
    yv = df.y.values.astype(float); yt = np.clip(yv[tr], *np.nanquantile(yv[tr], TARGET_CLIP))
    X = features(df, num, cats, seed=SEED)
    ret = df.se_nm.values[~tr] == "소매"; y = yv[~tr][ret]
    out = {}
    numc = [c for c in X.columns if c not in cats]
    Xn = X[numc].astype(float).mask(X[numc] <= -1e8); Xn = Xn.fillna(Xn[tr].median())
    Xn = pd.DataFrame(MinMaxScaler().fit(Xn[tr]).transform(Xn), columns=numc, index=Xn.index)
    Xl = pd.concat([Xn, pd.get_dummies(X[[c for c in cats if c != "item_cd"]].astype(str), dtype=float)], axis=1)
    out["선형회귀 (Ridge)"] = Ridge(alpha=1.0).fit(Xl[tr], yt).predict(Xl[~tr])[ret]
    for lab, cols in [("LightGBM · 가격 이력만", [c for c in X.columns if c in cats or c in PRICE or c.startswith("nat_")]), ("LightGBM · + 외부 변수", list(X.columns))]:
        out[lab] = lgb.LGBMRegressor(**lgb_params(d, cols)).fit(X.loc[tr, cols], yt, categorical_feature=[c for c in cats if c in cols]).predict(X.loc[~tr, cols])[ret]
    for k, p in out.items():
        m = metrics(y, p)
        rows.append({"분야": nm, "모델": k, "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4), "R²": round(m["R²"], 3), "방향 정확도": round(m["방향 정확도"], 3)})
    print(nm, "완료", flush=True)
r = pd.DataFrame(rows); r.to_csv(ROOT / "eda" / "tables" / "val_first_models.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200); print(r.to_string(index=False))
