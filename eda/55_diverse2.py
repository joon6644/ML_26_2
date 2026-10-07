# 6라운드: 스플라인 Ridge(GAM형) + MLP(참고용, 채택 기준: 시드 3개 모두 2σ 이상 개선)를 분야별 확정 구성에 더한 다양성 앙상블. 시드 42·0·1 (확인용), 예측 저장해 재사용
# 인자: 분야별 확정 구성 JSON 예) '{"agri": ["lgb","xgb","cat"], "livestock": ["lgb","xgb","cat","ridge"], "fishery": ["lgb"]}'
import importlib.util, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import Ridge
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "50_diverse.py", encoding="utf-8").read().split("for d in DOMS:")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
BASECFG = json.loads(sys.argv[1])
DOMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["livestock", "fishery", "agri"]
SEEDS = [42, 0, 1]
MAXTR = 1_500_000
SIG2 = {"농산물": 0.001, "축산물": 0.0004, "수산물": 0.0008}
rows = []


def dense_inputs(df, num, cats, fit_m):
    X = dv.matrix(df, num, cats, fit_m, 42)
    numX = X[[c for c in X.columns if c not in cats]].values.astype("float32")
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=200, sparse_output=False, dtype=np.float32).fit(df.loc[fit_m, cats].astype(str))
    return numX, oh


def mlp_pred(df, d, num, cats, tr, te, seed):
    numX, oh = dense_inputs(df, num, cats, tr)
    idx = np.where(tr)[0]
    if len(idx) > MAXTR: idx = np.random.default_rng(seed).choice(idx, MAXTR, replace=False)
    Z = lambda ix: np.hstack([np.clip(numX[ix], -1, 2), oh.transform(df.iloc[ix][cats].astype(str))])
    m = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, batch_size=2048, learning_rate_init=1e-3, max_iter=30,
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=4, random_state=seed)
    m.fit(Z(idx), ytrain(d, df.y.values.astype(float), tr)[np.searchsorted(np.where(tr)[0], idx)])
    te_ix = np.where(te)[0]
    return np.concatenate([m.predict(Z(te_ix[i:i + 200_000])) for i in range(0, len(te_ix), 200_000)])


def spline_pred(df, d, num, cats, tr, te):
    numX, oh = dense_inputs(df, num, cats, tr)
    idx = np.where(tr)[0]
    if len(idx) > MAXTR: idx = np.random.default_rng(42).choice(idx, MAXTR, replace=False)
    sp = SplineTransformer(n_knots=5, degree=3).fit(np.clip(numX[idx], -1, 2))
    Z = lambda ix: sparse.hstack([sparse.csr_matrix(sp.transform(np.clip(numX[ix], -1, 2)).astype(np.float32)), sparse.csr_matrix(oh.transform(df.iloc[ix][cats].astype(str)))]).tocsr()
    m = Ridge(alpha=10.0).fit(Z(idx), ytrain(d, df.y.values.astype(float), tr)[np.searchsorted(np.where(tr)[0], idx)])
    te_ix = np.where(te)[0]
    return np.concatenate([m.predict(Z(te_ix[i:i + 200_000])) for i in range(0, len(te_ix), 200_000)])


for d in DOMS:
    nm = dv.cs.rm.NAMES[d]
    df, num, cats = dv.cached_final_frame(d); tr, te = dv.masks(df); y = df.y.values[te].astype(float)
    base = BASECFG[d]
    t = time.time(); sp = spline_pred(df, d, num, cats, tr, te); t_sp = time.time() - t
    ridge_p = fit_all(df, d, num, cats, tr, te, models=["ridge"])[0]["ridge"] if "ridge" in base else None
    res = {}
    for s in SEEDS:
        SEED = s
        P = fit_all(df, d, num, cats, tr, te, models=[m for m in base if m != "ridge"])[0]
        if ridge_p is not None: P["ridge"] = ridge_p
        t = time.time(); P["mlp"] = mlp_pred(df, d, num, cats, tr, te, s); t_mlp = time.time() - t
        P["spline"] = sp
        np.savez(ROOT / "eda" / "tables" / f"preds_{d}_s{s}.npz", **P)
        cands = {"확정 구성": base, "+ 스플라인": base + ["spline"], "+ MLP (참고)": base + ["mlp"], "+ MLP + 스플라인 (참고)": base + ["mlp", "spline"],
                 "스플라인 단독": ["spline"], "MLP 단독 (참고)": ["mlp"]}
        for c, ms in cands.items():
            res.setdefault(c, []).append(dv.metrics(y, np.mean([P[m] for m in ms], axis=0)))
        err = {k: v - y for k, v in P.items()}
        print(nm, "시드", s, {c: round(res[c][-1]["RMSE"], 5) for c in cands}, "| LGB와 오차 상관 스플라인", round(np.corrcoef(err["spline"], err["lgb"])[0, 1], 3), "MLP", round(np.corrcoef(err["mlp"], err["lgb"])[0, 1], 3),
              f"| 스플라인 {t_sp:.0f}초, MLP 학습+추론 {t_mlp:.0f}초", flush=True)
    b = np.array([r["RMSE"] for r in res["확정 구성"]])
    for c, R in res.items():
        rm = np.array([r["RMSE"] for r in R]); diff = rm - b
        rows.append({"분야": nm, "구성": c, "RMSE 평균": round(rm.mean(), 5), "R² 평균": round(np.mean([r["R2"] for r in R]), 4),
                     "방향정확도": round(np.mean([r["방향정확도"] for r in R]), 4), "확정 구성 대비 ΔRMSE": f"{diff.mean():+.5f} ({(diff < 0).sum()}/{len(diff)} 개선)",
                     "판정": "-" if c == "확정 구성" else (("채택 기준 충족 (참고)" if (diff < -SIG2[nm]).all() else "기준 미달 (참고)") if "MLP" in c else
                                                        ("좋아짐" if diff.mean() < -SIG2[nm] else ("나빠짐" if diff.mean() > SIG2[nm] else "차이 없음")))})
    pd.DataFrame(rows).to_csv(ROOT / "eda" / "tables" / "diverse2.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([r for r in rows if r["분야"] == nm]).to_string(index=False), flush=True)
    del df
