# 결측 처리 비교 (사용자 요청 2026-10-09): 학습 구간 평균으로 채움(현재) vs 아무 처리 안 함(NaN 그대로)
# 나머지는 확정 구성 v2와 동일: 정제 데이터로 학습, 타깃 1~99% 자르기, LGB+XGB+Cat 평균, 단조 제약(축·수), 시드 42, MinMax(학습 구간, NaN은 건너뜀)
# 세 모델 모두 결측을 스스로 다룸: LightGBM·XGBoost = 갈라질 때 NaN을 보낼 쪽을 학습 / CatBoost = NaN을 가장 작은 값으로 취급(기본)
# 검증 = processed_R/model_frame_v1 (~2022-12-01) → 원래 model_frame_v1 (2023~24) · 테스트 = model_input.parquet (학습 → 테스트)
# 비교 기준 = eda/tables/final_eval.csv (평균 채움) · 출력 eda/tables/no_impute.csv
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
T, P0, PR = ROOT / "eda" / "tables", ROOT / "data" / "processed", ROOT / "data" / "processed_R"
NAMES = dv.cs.rm.NAMES
lab = lambda v: np.where(v > .01, 1, np.where(v < -.01, -1, 0))


def prep_nofill(df, num, tr_m):
    """결측을 채우지 않는 전처리: 무한대만 NaN으로, MinMax는 학습 구간 기준 (NaN은 건너뛰고 그대로 둠)."""
    X = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X).astype("float32"), columns=num, index=df.index)


dv.cs.rm.prep = prep_nofill                                              # dv.matrix 가 쓰는 전처리를 교체


def run(d, R, E, num):
    df = pd.concat([R, E[R.columns]], ignore_index=True)
    df.attrs["scope"] = {c[8:]: df[c].values.astype(bool) for c in df.columns if c.startswith("_scope__")}
    tr = np.r_[np.ones(len(R), bool), np.zeros(len(E), bool)]
    yv = df.y.values.astype(float)
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], (0.01, 0.99)))
    dv.TRAIN_CFG[d] = {"clip": (0.01, 0.99), "params": {}}
    mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
    return fit3(df, d, num, dv.base_sets(NAMES[d])[1], tr, ~tr, yt, 42, mono)


rows = []
for d in ["livestock", "fishery", "agri"]:
    nm = NAMES[d]
    num = json.loads((P0 / d / "model_input.json").read_text(encoding="utf-8"))["num"]
    for sname in ["검증 (2023~24)", "테스트 (2025~26)"]:
        t = time.time()
        f = T / f"noimpute_preds_{d}_{'val' if sname.startswith('검증') else 'test'}.npz"
        if sname.startswith("검증"):
            O = pd.read_parquet(P0 / d / "model_frame_v1.parquet")
            E = O[(O.date >= "2023-01-01") & (O.date <= "2024-12-31") & O.y.notna()]
            R = pd.read_parquet(PR / d / "model_frame_v1.parquet")
            R = R[(R.date >= "2016-01-01") & (R.date <= "2022-12-01") & R.y.notna()]
        else:
            M = pd.read_parquet(P0 / d / "model_input.parquet")
            R, E = M[M.split == "학습"].drop(columns="split"), M[M.split == "테스트"].drop(columns="split")
        P = dict(np.load(f)) if f.exists() else run(d, R, E, num)
        np.savez(f, **P)
        r = (E.se_nm == "소매").values
        y = E.y.values.astype(float)[r]
        P3 = np.column_stack([P[k] for k in ["lgb", "xgb", "cat"]])[r]
        p = P3.mean(1)
        for model, pp in [("LightGBM", P3[:, 0]), ("XGBoost", P3[:, 1]), ("CatBoost", P3[:, 2]), ("최종 (평균)", p)]:
            m = dv.metrics(y, pp)
            rows.append({"분야": nm, "구간": sname, "결측 처리": "없음 (NaN 그대로)", "모델": model, "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4),
                         "R²": round(m["R2"], 3), "방향 정확도": round(m["방향정확도"], 3)})
        L = np.column_stack([lab(P3[:, i]) for i in range(3)]); hi = (L == L[:, :1]).all(1) & (L[:, 0] != 0); mv = np.abs(y) > .01
        rows[-1].update({"높음 비율": round(float(hi.mean()), 3), "높음 방향 정확도": round(float(np.mean(np.sign(p[hi & mv]) == np.sign(y[hi & mv]))), 3)})
        print(nm, sname, f"{time.time() - t:.0f}초", rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(T / "no_impute.csv", index=False, encoding="utf-8-sig")
        del R, E

# 평균 채움(현재 v2)과 나란히
base = pd.read_csv(T / "final_eval.csv")
base = base[base.모델 == "최종 (평균)"][["분야", "구간", "RMSE", "R²", "방향 정확도"]].assign(**{"결측 처리": "학습 구간 평균 (현재)"})
new = pd.DataFrame(rows)
new = new[new.모델 == "최종 (평균)"][["분야", "구간", "RMSE", "R²", "방향 정확도", "결측 처리", "높음 비율", "높음 방향 정확도"]]
pd.set_option("display.width", 200)
print(pd.concat([base, new]).sort_values(["분야", "구간", "결측 처리"]).to_string(index=False))
