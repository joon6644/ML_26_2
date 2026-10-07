# 테스트셋 1회 평가 (사용자 요청 2026-10-07): 2016-01-01 ~ 2024-12-01 학습 → 2025-01-01 ~ 끝(타깃 있는 날까지) 예측
# 확정 구성(LGB+XGB+Cat 균등 평균, 분야별 학습 설정·단조 제약, 시드 42) 그대로. 설정은 바꾸지 않음
# 지표(2026-10-07 확정): 베이스라인 + 모델, RMSE·MAE·R², 방향 정확도(|y|>1%, 평균 예측 부호). 보조: 신뢰도(±1% 세 모델 만장일치)별·변화 크기별
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
SPLIT = ("2016-01-01", "2024-12-01", "2025-01-01", "2026-12-31")
dv.cs.rm.SPLIT = dv.cs.SPLIT = dv.cs.fs.SPLIT = SPLIT   # 학습 구간 정규화(복합 지표·MinMax)도 학습 구간 기준으로
OUT = ROOT / "eda" / "tables"
BAND = dv.BAND
c1 = lambda v: np.where(v > BAND, 1, np.where(v < -BAND, -1, 0))


def frame(d):
    path = dv.cs.P / d / "model_frame_test.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        sc = [c for c in df.columns if c.startswith("_scope__")]
        df.attrs["scope"] = {c[8:]: df[c].values.astype(bool) for c in sc}
        num = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["num"]
        return df.drop(columns=sc), num
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
    path.with_suffix(".json").write_text(json.dumps({"num": num}, ensure_ascii=False), encoding="utf-8")
    return frame(d)


rows, aux, miss = [], [], []
for d in ["livestock", "fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]; t = time.time()
    df, num = frame(d); cats = dv.base_sets(nm)[1]
    tr = ((df.date >= SPLIT[0]) & (df.date <= SPLIT[1])).values
    te = ((df.date >= SPLIT[2]) & df.y.notna()).values
    va = ((df.date >= "2023-01-01") & (df.date <= "2024-12-31")).values
    # 데이터 공백 점검: 테스트 구간에 새로 비는 피처 (수집 중단 등)
    for c in num:
        a, b = df.loc[va, c].isna().mean(), df.loc[te, c].isna().mean()
        if b - a > 0.10:
            miss.append({"분야": nm, "피처": c, "검증 결측": round(a, 3), "테스트 결측": round(b, 3)})
    print(nm, "테스트 기간", df.date[te].min().date(), "~", df.date[te].max().date(), "행", int(te.sum()), "/ 공백 피처", [m["피처"] for m in miss if m["분야"] == nm], flush=True)
    f = OUT / f"test_preds_{d}.npz"
    if f.exists():
        P = dict(np.load(f))
    else:
        cfg = dv.TRAIN_CFG[d]; yv = df.y.values.astype(float)
        yt = np.clip(yv[tr], *np.nanquantile(yv[tr], cfg["clip"])) if cfg.get("clip") else yv[tr]
        mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
        P = fit3(df, d, num, cats, tr, te, yt, 42, mono)
        np.savez(f, **P)
    # 정해진 형식: 분야별 베이스라인 + 모델, RMSE·MAE·R² + 방향 정확도(|y|>1%). 보조(최종 모델): 신뢰도별·변화 크기별 방향 정확도
    y = df.y.values[te].astype(float); Pm = np.column_stack([P[k] for k in ["lgb", "xgb", "cat"]]); p = Pm.mean(1); mv = np.abs(y) > BAND
    for model, pp in [("베이스라인 (최근 7일 평균 = 향후 4주 평균)", np.zeros(len(y))),
                      ("LightGBM", P["lgb"]), ("XGBoost", P["xgb"]), ("CatBoost", P["cat"]), ("최종 (LGB+XGB+Cat 평균)", p)]:
        r = dv.metrics(y, pp)
        rows.append({"분야": nm, "모델": model, "RMSE": round(r["RMSE"], 5), "MAE": round(r["MAE"], 5), "R²": round(r["R2"], 4),
                     "방향 정확도": round(r["방향정확도"], 4) if pp.any() else np.nan})
    ok = np.sign(p) == np.sign(y); L = c1(Pm); ag = (L == L[:, :1]).all(1)
    for tier, m in [("높음 (세 모델 모두 +1% 초과 또는 −1% 미만)", ag & (L[:, 0] != 0)), ("보통 (세 모델 모두 ±1% 이내)", ag & (L[:, 0] == 0)), ("낮음 (엇갈림)", ~ag)]:
        aux.append({"분야": nm, "구분": "신뢰도", "항목": tier, "행 비율": round(float(m.mean()), 3), "방향 정확도": round(float(ok[m & mv].mean()), 4)})
    for lo_, hi_, lab in [(.01, .02, "1~2%"), (.02, .05, "2~5%"), (.05, .1, "5~10%"), (.1, 99, "10% 이상")]:
        m = (np.abs(y) > lo_) & (np.abs(y) <= hi_)
        aux.append({"분야": nm, "구분": "실제 변화 크기", "항목": lab, "행 비율": round(float(m.mean()), 3), "방향 정확도": round(float(ok[m].mean()), 4)})
    print(nm, "초", round(time.time() - t), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "test_eval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aux).to_csv(OUT / "test_eval_aux.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(miss).to_csv(OUT / "test_missing_features.csv", index=False, encoding="utf-8-sig")
    del df
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(aux).to_string(index=False))
print(pd.DataFrame(miss).to_string(index=False))
