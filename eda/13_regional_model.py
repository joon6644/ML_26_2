# %% [markdown]
# # 지역 단위 학습 (향후 28일 평균 가격, 지역별 시계열)
# 시계열 = 품목 × 품종 × 등급 × 소매/중도매 × 지역(sgg). 지역 가격 = 그날 그 지역 조사 시장 가격의 중앙값
# 타깃 y = 지역 향후 28일 평균 / 지역 최근 7일 평균 − 1
# 비교 (같은 지역 테스트 행에서 평가):
#   - 지역 모델: 지역 가격 이력 + 전국 가격 이력 + 지역 프리미엄 + 외부 변수(전국 피처와 동일) + 지역 범주
#   - 전국 모델 적용: 기존 전국 모델(features.parquet로 학습)의 예측값을 그 날짜·시계열의 모든 지역에 그대로 사용
# 학습 2016~2022 / 테스트 2023~2024, 전처리: 결측 = 학습셋 평균, MinMax(학습셋 fit)
# 실행: `python eda/13_regional_model.py` → eda/13_regional_model.md

# %%
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_features as bf  # noqa: E402

P = ROOT / "data" / "processed"
H = 28
SPLIT = ("2016-01-01", "2022-12-01", "2023-01-01", "2024-12-31")
KEYS = ["ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
RKEYS = KEYS + ["sgg_nm"]
CAT = ["item_cd", "food_group", "se_nm", "sgg_nm"]
LIN_CAT = ["item_cd", "se_nm", "sgg_nm"]
HIST = bf.HIST
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}


def regional_rows(d):
    p = pd.read_parquet(P / d / "price.parquet", columns=["date", "se_nm", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "sgg_nm", "price_kg"])
    p = p[p.se_nm.isin(["소매", "중도매"]) & (p.price_kg > 0) & (p.sgg_nm != "전국") & (p.date >= "2015-01-01")]
    daily = p.groupby(RKEYS + ["date"], observed=True).price_kg.median().reset_index()
    parts = []
    for _, g in daily.groupby(RKEYS, observed=True, sort=False):
        s = g.set_index("date").price_kg.asfreq("D")
        base = bf.rmean(s, 0, 6, 3)
        f = pd.DataFrame(index=s.index)
        f["base"] = base
        f["r7"] = np.log(base / bf.rmean(s, 7, 13, 3))
        f["r28"] = np.log(base / bf.rmean(s, 28, 34, 3))
        f["r91"] = np.log(base / bf.rmean(s, 91, 97, 3))
        f["yoy"] = np.log(base / bf.rmean(s, 358, 371, 3))
        f["vol28"] = np.log(s).diff().rolling(28, min_periods=10).std()
        f["dev_last"] = np.log(s / base)
        f["ly_change"] = np.log(s.shift(364 - H).rolling(H, min_periods=10).mean() / bf.rmean(s, 358, 364, 3))
        f["n_obs28"] = s.notna().rolling(28, min_periods=1).sum()
        fut = s[::-1].rolling(H, min_periods=max(5, H // 3)).mean()[::-1].shift(-1)
        f["y"] = fut / base - 1
        parts.append(f.reindex(g.date).set_index(g.index))
    out = daily.join(pd.concat(parts))
    out[HIST] = out[HIST].replace([np.inf, -np.inf], np.nan)
    t0, t1, s0, s1 = SPLIT
    keep = out.y.notna() & (((out.date >= t0) & (out.date <= t1)) | ((out.date >= s0) & (out.date <= s1)))
    return out[keep].reset_index(drop=True)


def national_rows(d):
    """전국 시계열: 외부 변수 + 전국 가격 이력 (모든 날짜, 지역 행에 붙이기용)."""
    ds = pd.read_parquet(P / d / "dataset.parquet")
    ds = bf.add_history_and_target(ds)
    ext = [c for c in bf.COMMON + bf.DOMAIN[d] if c not in ("price_kg",)]
    nat = ds[KEYS + ["date", "food_group", "base"] + HIST + ext].rename(columns={"base": "nat_base", **{h: f"nat_{h}" for h in HIST}})
    return nat, ext


def prep(df, num, tr_m):
    X = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr_m].mean()).fillna(0)
    return pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X).astype("float32"), columns=num, index=df.index)


def with_cat(Xs, df):
    Xc = Xs.copy()
    for c in CAT:
        if c in df:
            Xc[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    return Xc


def metrics(y, yhat):
    return {"RMSE": float(np.sqrt(np.mean((yhat - y) ** 2))), "MAE": float(np.mean(np.abs(yhat - y))),
            "R2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))}


XGB = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
           tree_method="hist", device="cuda", enable_categorical=True, random_state=0)
LGB = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, random_state=0, verbose=-1)


def national_model_preds(d):
    """기존 확정 포맷의 전국 모델 (features.parquet 전체 피처) → 전국 행 예측값."""
    f = pd.read_parquet(P / d / "features.parquet")
    t0, t1, s0, s1 = SPLIT
    num = [c for c in f.columns if c not in bf.ID + ["base", "y"]]
    tr_m = (f.date >= t0) & (f.date <= t1)
    te_m = (f.date >= s0) & (f.date <= s1)
    Xc = with_cat(prep(f, num, tr_m), f)
    cats = [c for c in CAT if c in Xc]
    out = f.loc[te_m, KEYS + ["date"]].copy()
    out["nat_xgb"] = xgb.XGBRegressor(**XGB).fit(Xc[tr_m], f.y[tr_m]).predict(Xc[te_m])
    out["nat_lgb"] = lgb.LGBMRegressor(**LGB).fit(Xc[tr_m], f.y[tr_m], categorical_feature=cats).predict(Xc[te_m])
    return out


def main():
    t_all = time.time()
    rows, info = [], []
    L = ["# 지역 단위 학습 결과 (향후 28일 평균 가격, 테스트 2023~2024)", "",
         "> 자동 생성: `python eda/13_regional_model.py`",
         "> 타깃 = 지역 향후 28일 평균 / 지역 최근 7일 평균 − 1. 모든 행은 지역 시계열의 테스트 행에서 평가",
         "> 지역 모델 = 지역 가격 이력 + 전국 가격 이력 + 지역 프리미엄(지역 7일 평균 / 전국 7일 평균 − 1) + 외부 변수 + 지역 범주",
         "> 전국 모델 적용 = 기존 전국 모델 예측값을 그 날짜·시계열의 모든 지역에 그대로 사용", ""]
    for d, nm in NAMES.items():
        t = time.time()
        reg = regional_rows(d)
        nat, ext = national_rows(d)
        df = reg.merge(nat, on=KEYS + ["date"], how="left")
        del reg, nat
        df["premium"] = df.base / df.nat_base - 1
        df["price_kg"] = df.price_kg.astype("float32")
        df["woy"] = df.date.dt.isocalendar().week.astype("int16")
        num = ["price_kg"] + HIST + ["woy"] + ext + [f"nat_{h}" for h in HIST] + ["premium"]
        t0, t1, s0, s1 = SPLIT
        tr_m = ((df.date >= t0) & (df.date <= t1)).values
        te_m = ((df.date >= s0) & (df.date <= s1)).values
        y = df.y.values.astype("float64")
        Xs = prep(df, num, tr_m)
        res = {"베이스라인 1 (지역 오늘 가격 = 향후 4주 평균)": metrics(y[te_m], np.expm1(df.dev_last.fillna(0).values[te_m]))}
        # 선형회귀 (지역)
        oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(df.loc[tr_m, LIN_CAT].astype(str))
        lx = lambda m: hstack([csr_matrix(Xs.values[m]), oh.transform(df.loc[m, LIN_CAT].astype(str))]).tocsr()
        res["지역 모델 · 선형회귀"] = metrics(y[te_m], LinearRegression().fit(lx(tr_m), y[tr_m]).predict(lx(te_m)))
        Xc = with_cat(Xs, df)
        del Xs
        res["지역 모델 · XGBoost"] = metrics(y[te_m], xgb.XGBRegressor(**XGB).fit(Xc[tr_m], y[tr_m]).predict(Xc[te_m]))
        lg = lgb.LGBMRegressor(**LGB).fit(Xc[tr_m], y[tr_m], categorical_feature=CAT)
        res["지역 모델 · LightGBM"] = metrics(y[te_m], lg.predict(Xc[te_m]))
        imp = pd.Series(lg.booster_.feature_importance("gain"), index=Xc.columns)
        del Xc
        # 전국 모델 예측을 지역 행에 적용
        npred = national_model_preds(d)
        te = df.loc[te_m, KEYS + ["date", "y"]].merge(npred, on=KEYS + ["date"], how="left")
        ok = te.nat_lgb.notna().values
        res["전국 모델 적용 · XGBoost"] = metrics(te.y.values[ok], te.nat_xgb.values[ok])
        res["전국 모델 적용 · LightGBM"] = metrics(te.y.values[ok], te.nat_lgb.values[ok])
        for k, v in res.items():
            rows.append({"분야": nm, "모델": k, **{a: round(b, 4) for a, b in v.items()}})
        top = (imp / imp.sum()).sort_values(ascending=False).head(10)
        info.append((nm, df.sgg_nm.nunique(), df.groupby(RKEYS).ngroups, int(tr_m.sum()), int(te_m.sum()), f"{ok.mean() * 100:.1f}%",
                     ", ".join(f"{k} {v:.2f}" for k, v in top.items()), round(time.time() - t)))
        print(nm, f"{time.time() - t:.0f}초", flush=True)
        print(pd.DataFrame([r for r in rows if r["분야"] == nm]).drop(columns="분야").to_string(index=False), flush=True)
        del df
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "eda" / "tables" / "regional_model.csv", encoding="utf-8-sig", index=False)
    for nm in NAMES.values():
        L += [f"## {nm}", "", r[r.분야 == nm].drop(columns="분야").to_markdown(index=False, disable_numparse=True), ""]
    L += ["## 데이터 · 중요 피처", "", pd.DataFrame(info, columns=["분야", "지역 수", "지역 시계열 수", "학습 행", "테스트 행",
                                                            "전국 예측 매칭률", "지역 LightGBM 상위 피처 (gain 비중)", "초"]).to_markdown(index=False), "",
          f"> 전체 소요 {time.time() - t_all:.0f}초"]
    (ROOT / "eda" / "13_regional_model.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
