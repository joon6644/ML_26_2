# [기록용 실험] 과거 설정으로 돌린 실험 — 현재 설정이 아님. 결과는 eda/derived_log.md, 현재 설정은 scripts/model_config.py, 학습은 demo/0_train_models.py (eda/README.md)
# ★ 확정 구성 v2 성능표 (2026-10-09): 데이터 정제(하루짜리 튐 → 전날 값) + 학습 처리(타깃 1~99% 자르기) + 결측 처리 없음, 세 분야 공통
# 학습 = 정제 데이터 (data/processed_R) / 평가 = 원래 데이터 · 소매 행 · 베이스라인 + LGB·XGB·Cat·최종(평균), RMSE·MAE·R²·방향 정확도(±1%)
#   검증: ~2022-12-01 학습 → 2023~24 (여기서 학습) · 테스트: ~2024-12-01 학습 → 2025~ (demo/0_train_models.py 가 저장한 예측)
# 출력: eda/tables/final_eval.csv (성능표), final_eval_aux.csv (신뢰도 등급별·실제 변화 크기별)
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
T = ROOT / "eda" / "tables"
P0, PR = ROOT / "data" / "processed", ROOT / "data" / "processed_R"
NAMES = dv.cs.rm.NAMES
from sklearn.preprocessing import MinMaxScaler   # noqa: E402


def prep_nofill(df, num, tr_m):
    """결측 처리 없음 (2026-10-09 채택, eda/71): 무한대만 NaN으로, MinMax는 학습 구간 기준 (NaN 그대로)."""
    X = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(MinMaxScaler().fit(X[tr_m]).transform(X).astype("float32"), columns=num, index=df.index)


dv.cs.rm.prep = prep_nofill
VAL_SPLIT = ("2016-01-01", "2022-12-01", "2023-01-01", "2024-12-31")
lab = lambda v: np.where(v > .01, 1, np.where(v < -.01, -1, 0))


def clean_v1(d):
    """검증용 정제 학습 표 (복합 지표 정규화를 2022년까지 기준으로 — 원래 model_frame_v1과 같은 규칙)."""
    path = PR / d / "model_frame_v1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    dv.cs.rm.P = dv.cs.P = dv.cs.fs.P = PR
    dv.cs.rm.SPLIT = dv.cs.SPLIT = dv.cs.fs.SPLIT = VAL_SPLIT
    df, num, cats = dv.final_frame(d)
    keep = sorted(set(dv.cs.KEYS + ["sgg_nm", "date", "y", "is_nat", "food_group", "dev_last", "base"] + num + cats))
    out = df[[c for c in keep if c in df.columns]].copy()
    for c, ok in df.attrs.get("scope", {}).items():
        out[f"_scope__{c}"] = ok
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.attrs = {}
    out.to_parquet(path, index=False)
    dv.cs.rm.P = dv.cs.P = dv.cs.fs.P = P0
    return out


def val_preds(d):
    f = T / f"final_val_preds_{d}.npz"
    O = pd.read_parquet(P0 / d / "model_frame_v1.parquet")
    E = O[(O.date >= VAL_SPLIT[2]) & (O.date <= VAL_SPLIT[3]) & O.y.notna()]
    if f.exists():
        return dict(np.load(f)), E
    R = clean_v1(d)
    R = R[(R.date >= VAL_SPLIT[0]) & (R.date <= VAL_SPLIT[1]) & R.y.notna()]
    num = json.loads((P0 / d / "model_frame_v1.json").read_text(encoding="utf-8"))["num"]
    df = pd.concat([R, E[R.columns]], ignore_index=True)
    df.attrs["scope"] = {c[8:]: df[c].values.astype(bool) for c in df.columns if c.startswith("_scope__")}
    tr = np.r_[np.ones(len(R), bool), np.zeros(len(E), bool)]
    yv = df.y.values.astype(float)
    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], (0.01, 0.99)))
    dv.TRAIN_CFG[d] = {"clip": (0.01, 0.99), "params": {}}            # 손실은 기본, 타깃만 자르기
    mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
    P = fit3(df, d, num, dv.base_sets(NAMES[d])[1], tr, ~tr, yt, 42, mono)
    np.savez(f, **P)
    return P, E


def test_preds(d):
    O = pd.read_parquet(P0 / d / "model_frame_test.parquet", columns=["date", "y", "se_nm"])
    E = O[(O.date >= "2025-01-01") & O.y.notna()]
    return dict(np.load(T / f"final_test_preds_{d}.npz")), E


rows, aux = [], []
for d in ["livestock", "fishery", "agri"]:
    nm = NAMES[d]
    for sname, get in [("검증 (2023~24)", val_preds), ("테스트 (2025~26)", test_preds)]:
        t = time.time()
        P, E = get(d)
        r = (E.se_nm == "소매").values
        y = E.y.values.astype(float)[r]
        P3 = np.column_stack([P[k] for k in ["lgb", "xgb", "cat"]])[r]
        p = P3.mean(1)
        for model, pp in [("베이스라인", np.zeros(len(y))), ("LightGBM", P3[:, 0]), ("XGBoost", P3[:, 1]), ("CatBoost", P3[:, 2]), ("최종 (평균)", p)]:
            m = dv.metrics(y, pp)
            rows.append({"분야": nm, "구간": sname, "모델": model, "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4), "R²": round(m["R2"], 3),
                         "방향 정확도": round(m["방향정확도"], 3) if pp.any() else np.nan})
        L = np.column_stack([lab(P3[:, i]) for i in range(3)]); ag = (L == L[:, :1]).all(1)
        ok, mv = np.sign(p) == np.sign(y), np.abs(y) > .01
        for tier, m in [("높음", ag & (L[:, 0] != 0)), ("보통", ag & (L[:, 0] == 0)), ("낮음", ~ag)]:
            aux.append({"분야": nm, "구간": sname, "구분": "신뢰도", "항목": tier, "비율": round(float(m.mean()), 3), "방향 정확도": round(float(ok[m & mv].mean()), 3)})
        for lo, hi, lb in [(.01, .02, "1~2%"), (.02, .05, "2~5%"), (.05, .1, "5~10%"), (.1, 99, "10% 이상")]:
            m = (np.abs(y) > lo) & (np.abs(y) <= hi)
            aux.append({"분야": nm, "구간": sname, "구분": "실제 변화 크기", "항목": lb, "비율": round(float(m.mean()), 3), "방향 정확도": round(float(ok[m].mean()), 3)})
        print(nm, sname, f"{time.time() - t:.0f}초", rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(T / "final_eval.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(aux).to_csv(T / "final_eval_aux.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).fillna("–").to_string(index=False))
print(pd.DataFrame(aux).to_string(index=False))
