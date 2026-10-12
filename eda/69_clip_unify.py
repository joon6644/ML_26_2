# [기록용 실험] 과거 설정으로 돌린 실험 — 현재 설정이 아님. 결과는 eda/derived_log.md, 현재 설정은 scripts/model_config.py, 학습은 demo/0_train_models.py (eda/README.md)
# 학습 처리 통일 확인 (사용자 요청 2026-10-09): 세 분야 모두 '학습 타깃 1~99% 자르기'로 통일하면?
# 축산은 이미 자르기 → 농산(현재 처리 없음)·수산(현재 Huber)만 자르기로 바꿔 세 모델 앙상블(시드 42)로 비교
# 검증 = model_frame_v1 (~2022-12-01 학습 → 2023~24), 테스트 = model_frame_test (~2024-12-01 학습 → 2025~). 현재 구성 예측은 저장된 것 사용
# 지표: 소매 행, RMSE·R²·방향 정확도(±1%)·높음(세 모델 ±1% 같은 방향) 비율과 방향 정확도
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
T = ROOT / "eda" / "tables"
P0 = ROOT / "data" / "processed"
CUR = {("agri", "검증"): "preds_agri_s42.npz", ("fishery", "검증"): "final_preds_fishery_s42.npz",
       ("agri", "테스트"): "test_preds_agri.npz", ("fishery", "테스트"): "test_preds_fishery.npz"}
SPL = {"검증": ("model_frame_v1", "2022-12-01", "2023-01-01", "2024-12-31"), "테스트": ("model_frame_test", "2024-12-01", "2025-01-01", "2099-12-31")}
lab = lambda v: np.where(v > .01, 1, np.where(v < -.01, -1, 0))


def score(y, P3):
    p = P3.mean(1)
    r = dv.metrics(y, p)
    L = np.column_stack([lab(P3[:, i]) for i in range(3)])
    hi = (L == L[:, :1]).all(1) & (L[:, 0] != 0)
    mv = np.abs(y) > .01
    return {"RMSE": round(r["RMSE"], 5), "R²": round(r["R2"], 4), "방향 정확도": round(r["방향정확도"], 4),
            "높음 비율": round(float(hi.mean()), 3), "높음 방향 정확도": round(float(np.mean(np.sign(p[hi & mv]) == np.sign(y[hi & mv]))), 4)}


rows = []
for d in ["fishery", "agri"]:
    nm = dv.cs.rm.NAMES[d]
    for sname, (fname, tend, es, ee) in SPL.items():
        t = time.time()
        df = pd.read_parquet(P0 / d / f"{fname}.parquet")
        num = json.loads((P0 / d / f"{fname}.json").read_text(encoding="utf-8"))["num"]
        df.attrs["scope"] = {c[8:]: df[c].values.astype(bool) for c in df.columns if c.startswith("_scope__")}
        cats = dv.base_sets(nm)[1]
        tr = ((df.date >= "2016-01-01") & (df.date <= tend)).values
        te = ((df.date >= es) & (df.date <= ee) & df.y.notna()).values
        y = df.y.values[te].astype(float)
        ret = df.se_nm.values[te] == "소매"
        z = np.load(T / CUR[(d, sname)])
        assert len(z["lgb"]) == te.sum(), (d, sname, len(z["lgb"]), te.sum())
        rows.append({"분야": nm, "구간": sname, "학습 처리": "현재 (" + ("없음" if d == "agri" else "Huber") + ")",
                     **score(y[ret], np.column_stack([z[k] for k in ["lgb", "xgb", "cat"]])[ret])})
        print(rows[-1], flush=True)
        f = T / f"clip_preds_{d}_{'val' if sname == '검증' else 'test'}.npz"
        if f.exists():
            P = dict(np.load(f))
        else:
            dv.TRAIN_CFG[d] = {"clip": (0.01, 0.99), "params": {}}           # 손실은 기본(제곱 오차), 타깃만 자르기
            yv = df.y.values.astype(float)
            yt = np.clip(yv[tr], *np.nanquantile(yv[tr], (0.01, 0.99)))
            mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
            P = fit3(df, d, num, cats, tr, te, yt, 42, mono)
            np.savez(f, **P)
        rows.append({"분야": nm, "구간": sname, "학습 처리": "1~99% 자르기",
                     **score(y[ret], np.column_stack([P[k] for k in ["lgb", "xgb", "cat"]])[ret]), "초": round(time.time() - t)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(T / "clip_unify.csv", index=False, encoding="utf-8-sig")
        del df
print(pd.DataFrame(rows).to_string(index=False))
