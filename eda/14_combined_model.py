# %% [markdown]
# # 지역 + 전국 통합 학습 (분야별 모델 1개로 지역 예측과 전국 예측을 함께)
# 학습 데이터 = 지역 시계열 행(13번과 동일) + 전국 시계열 행(features.parquet, 지역 = "전국", 프리미엄 = 0, 전국 가격 이력 = 자기 이력)
# 평가: 같은 모델을 지역 테스트 행 / 전국 테스트 행에서 따로 측정
# 비교 대상: 지역 전용 모델(13번), 전국 전용 모델(확정 포맷, features.parquet)
# 학습 2016~2022 / 테스트 2023~2024
# 실행: `python eda/14_combined_model.py` → eda/14_combined_model.md

# %%
import importlib.util
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rm", ROOT / "eda" / "13_regional_model.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
P, KEYS, HIST, SPLIT, CAT = rm.P, rm.KEYS, rm.HIST, rm.SPLIT, rm.CAT


def build(d):
    reg = rm.regional_rows(d)
    nat, ext = rm.national_rows(d)
    df = reg.merge(nat, on=KEYS + ["date"], how="left")
    del reg
    df["premium"] = df.base / df.nat_base - 1
    f = pd.read_parquet(P / d / "features.parquet")
    t0, t1, s0, s1 = SPLIT
    f = f[((f.date >= t0) & (f.date <= t1)) | ((f.date >= s0) & (f.date <= s1))].copy()
    f["sgg_nm"] = "전국"
    for h in HIST:
        f[f"nat_{h}"] = f[h]
    f["premium"] = 0.0
    f["nat_base"] = f.base
    cols = [c for c in df.columns if c in f.columns]
    df = pd.concat([df[cols].assign(is_nat=0), f[cols].assign(is_nat=1)], ignore_index=True)
    df["woy"] = df.date.dt.isocalendar().week.astype("int16")
    num = ["price_kg"] + HIST + ["woy"] + ext + [f"nat_{h}" for h in HIST] + ["premium"]
    return df, num


def main():
    t_all = time.time()
    rows = []
    for d, nm in rm.NAMES.items():
        t = time.time()
        df, num = build(d)
        t0, t1, s0, s1 = SPLIT
        tr_m = ((df.date >= t0) & (df.date <= t1)).values
        te_m = ((df.date >= s0) & (df.date <= s1)).values
        y = df.y.values.astype("float64")
        Xc = rm.with_cat(rm.prep(df, num, tr_m), df)
        p_x = xgb.XGBRegressor(**rm.XGB).fit(Xc[tr_m], y[tr_m]).predict(Xc[te_m])
        p_l = lgb.LGBMRegressor(**rm.LGB).fit(Xc[tr_m], y[tr_m], categorical_feature=CAT).predict(Xc[te_m])
        nat_te = df.is_nat.values[te_m] == 1
        yt, dev = y[te_m], np.expm1(df.dev_last.fillna(0).values[te_m])
        for part, m in [("지역 테스트 행", ~nat_te), ("전국 테스트 행", nat_te)]:
            for model, pred in [("베이스라인 1 (오늘 가격 = 향후 4주 평균)", dev), ("통합 모델 · XGBoost", p_x), ("통합 모델 · LightGBM", p_l)]:
                rows.append({"분야": nm, "평가 대상": part, "모델": model, **{k: round(v, 4) for k, v in rm.metrics(yt[m], pred[m]).items()}})
        print(nm, f"{time.time() - t:.0f}초", flush=True)
        print(pd.DataFrame([r for r in rows if r["분야"] == nm]).drop(columns="분야").to_string(index=False), flush=True)
        del df, Xc
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "eda" / "tables" / "combined_model.csv", encoding="utf-8-sig", index=False)
    L = ["# 지역 + 전국 통합 학습 (테스트 2023~2024)", "", "> 자동 생성: `python eda/14_combined_model.py`",
         "> 분야별 모델 1개를 지역 행 + 전국 행(지역 = '전국')으로 함께 학습하고, 지역 행과 전국 행에서 따로 평가", ""]
    for nm in rm.NAMES.values():
        L += [f"## {nm}", "", r[r.분야 == nm].drop(columns="분야").to_markdown(index=False, disable_numparse=True), ""]
    L.append(f"> 전체 소요 {time.time() - t_all:.0f}초")
    (ROOT / "eda" / "14_combined_model.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
