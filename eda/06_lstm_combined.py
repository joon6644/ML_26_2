# %% [markdown]
# # LSTM: 농·축·수산 통합 학습 (향후 30일 평균 가격)
# 05와 같은 피처·전처리·분할(학습 2016~2022, 테스트 2023~2024). 세 분야를 합쳐 LSTM 하나로 학습하고 분야별로 평가.
# 분야 전용 피처는 다른 분야 행에서 결측 → 학습셋 평균으로 채우고, 분야 원핫 3개를 피처로 추가.
# 실행: `python eda/06_lstm_combined.py` → eda/06_lstm_combined.md

# %%
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("m30", ROOT / "eda" / "05_models_h30_2023_24.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NAMES = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}


def main():
    t0 = time.time()
    common = m.bm.common_features()
    item_map = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str)[["ctgry_cd", "item_cd", "food_group"]]
    frames, num = [], []
    offset = 0
    for d in NAMES:
        df, cols = m.load_domain(d, common, item_map)
        df["domain"] = d
        df["sid"] += offset                                         # 시계열 id가 분야끼리 겹치지 않게
        offset = df.sid.max() + 1
        frames.append(df)
        num += [c for c in cols if c not in num]
    df = pd.concat(frames, ignore_index=True)
    for d in NAMES:
        df[f"dom_{d}"] = (df.domain == d).astype("float32")
    num_all = num + [f"dom_{d}" for d in NAMES]
    df = df.sort_values(["sid", "date"]).reset_index(drop=True)

    ok = df.y.notna() & df.base.notna()
    tr_m = ok & (df.date >= m.TRAIN_START) & (df.date <= m.TRAIN_END)
    te_m = ok & (df.date >= m.TEST_START) & (df.date <= m.TEST_END)
    Xs = m.preprocess(df, num_all)                                  # 결측 = 학습셋 평균, MinMax(학습셋 fit)
    y = df.y.values.astype("float32")
    va = np.where(tr_m & (df.date >= m.VAL_START))[0]
    tr_l = np.where(tr_m & (df.date < pd.Timestamp(m.VAL_START) - pd.Timedelta(days=m.H)))[0]
    te = np.where(te_m)[0]
    item_idx = (df.ctgry_cd + "_" + df.item_cd).astype("category").cat.codes.values.astype("int64")
    t = time.time()
    pred, n_ep, hist = m.train_lstm(df, Xs, y, tr_l, va, te, item_idx)
    secs = time.time() - t

    rows, lines = [], []
    te_df = df.iloc[te].assign(pred=pred)
    for d, nm in NAMES.items():
        sub = te_df[te_df.domain == d]
        sep = pd.read_parquet(m.P / "baseline_preds" / f"m30_{d}.parquet")      # 05의 분야별 결과 (같은 테스트 행)
        key = ["date", "item_nm", "vrty_nm", "grd_nm", "se_nm"]
        j = sep.merge(sub[key + ["pred"]], on=key, how="inner")
        res = {}
        for name, col in [("기준: 오늘 가격 유지", "pred_기준: 오늘 가격 유지"), ("선형회귀(분야별)", "pred_선형회귀"),
                          ("XGBoost(분야별)", "pred_XGBoost"), ("LSTM(분야별)", "pred_LSTM"), ("LSTM(통합)", "pred")]:
            res[name] = m.metrics(j.y.values, j[col].values.astype("float64"), j.base.values)
        r = pd.DataFrame(res).T
        r["오늘가격유지 대비 RMSE 개선(%)"] = (1 - r.RMSE / r.loc["기준: 오늘 가격 유지", "RMSE"]) * 100
        rows.append(r.assign(도메인=nm))
        print(nm, f"테스트 {len(j):,}행 (분야별 결과와 매칭)")
    s = pd.concat(rows).reset_index(names="모델")
    s.to_csv(m.TAB / "lstm_combined.csv", encoding="utf-8-sig", index=False)
    lines = ["# LSTM 농·축·수산 통합 학습 (향후 30일 평균 가격)", "",
             "> 자동 생성: `python eda/06_lstm_combined.py` · 분할·피처·전처리는 05와 동일 (테스트 2023~2024)",
             f"> 통합 학습 행 {len(tr_l):,} (검증 {len(va):,}), 피처 {len(num_all)}개 (분야 원핫 3개 포함), 품목 임베딩 {item_idx.max() + 1}개, "
             f"에폭 {n_ep}, 학습 {secs:.0f}초, 장치 {m.DEV}",
             f"> 검증 RMSE 추이: {' → '.join(f'{h:.4f}' for h in hist)}", "",
             s[["도메인", "모델", "RMSE", "MAE", "R2", "MAPE(%)", "방향정확도(%)", "급변 방향(%)", "급변 크기비", "오늘가격유지 대비 RMSE 개선(%)"]]
             .round(4).to_markdown(index=False, disable_numparse=True), ""]
    (ROOT / "eda" / "06_lstm_combined.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"학습 {len(tr_l):,}행, 에폭 {n_ep}, 학습 {secs:.0f}초, 전체 {time.time() - t0:.0f}초")
    print("검증 RMSE:", " → ".join(f"{h:.4f}" for h in hist))
    print(s[["도메인", "모델", "RMSE", "MAE", "R2", "방향정확도(%)", "급변 크기비"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
