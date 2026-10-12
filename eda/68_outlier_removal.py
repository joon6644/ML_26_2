# [파이프라인 모듈] 학습 데이터(model_frame·processed_R)를 만들 때 실행된다 — 지우지 말 것. 이 파일에 남은 학습 설정(전처리·모델·하이퍼파라미터)은 과거 실험값이며, 현재 서비스 설정은 scripts/model_config.py (eda/README.md)
# 이상치 제거 실험 (사용자 요청 2026-10-09): 원본 가격에서 이상치를 걸러 학습 데이터를 다시 만들면 더 잘 맞히나
#   A: 하루짜리 오류 — 직전 조사일 대비 1.5배 이상 튀었다가 다음 조사일에 원래 수준(±20%)으로 돌아온 날
#   R: A와 같은 하루짜리 튐을 지우지 않고 직전 조사일 가격으로 대치 (가설: 잘못 기록됐을 가능성이 높으니 가격은 전날 그대로였다)
#   B: 1.5 IQR 밖 — 직전 조사일 대비 가격 변화가 분야별 1.5 IQR 밖인 날 (울타리는 2016~2022 변화로 계산)
# 이상치 판정은 시장별 일별 가격(price.parquet)에서 하고, 걸러낸 가격으로 데이터셋 → 피처 → 학습 표를 data/processed_{A,B}/ 에 다시 만든다 (원본은 그대로).
# 걸러낸 데이터는 학습 구간에만 쓰고, 평가는 원래 데이터(원래 피처·원래 정답)로 한다.
# 학습 처리(없음 / 타깃 1~99% 자르기 / Huber)와 모두 조합해, 세 분야를 한 방식으로 통일할 수 있는지 본다.
# 탐색: LightGBM 단독, 시드 42 · 지표: 소매 행, R²·RMSE·방향 정확도(±1%) · 검증(~2022 학습 → 2023~24)과 테스트(~2024 학습 → 2025~)
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
src = open(ROOT / "eda" / "60_mono_scope.py", encoding="utf-8").read().split("for d in [")[0]
exec(src)
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb   # noqa: E402
import build_dataset as bd   # noqa: E402
import build_features as bfe   # noqa: E402
NAMES = dv.cs.rm.NAMES
_asof, _day = bd.asof, bd.day_merge   # 날짜 정밀도(us/ns)가 섞여도 병합되게 (원본 스크립트는 수정하지 않음)
bd.asof = lambda df, feat, by, cols: _asof(df, feat.assign(avail=feat["avail"].astype(df["date"].dtype)) if len(feat) else feat, by, cols)
bd.day_merge = lambda df, feat, on: _day(df, feat.assign(date=feat["date"].astype(df["date"].dtype)) if "date" in feat else feat, on)

P0 = ROOT / "data" / "processed"
SPLIT = ("2016-01-01", "2024-12-01", "2025-01-01", "2026-12-31")
SPLITS = {"검증": ("2022-12-01", "2023-01-01", "2024-12-31"), "테스트": ("2024-12-01", "2025-01-01", "2099-12-31")}
KM = ["se_nm", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "sgg_nm", "mrkt_cd"]
REBUILT = {"price.parquet", "dataset.parquet", "features.parquet"}
OUT = ROOT / "eda" / "tables" / "outlier_removal.csv"


def flags(p, how, fences=None):
    """시장별 일별 가격에서 이상치인 (시장, 날짜) 표시."""
    k = p.groupby(KM + ["date"], observed=True, dropna=False).price_kg.median().reset_index().sort_values(KM + ["date"])
    lp = np.log(k.price_kg)
    g = lp.groupby([k[c] for c in KM], observed=True, dropna=False)
    gap = (k.date - k.date.groupby([k[c] for c in KM], observed=True, dropna=False).shift(1)).dt.days
    d, nxt, prev = lp - g.shift(1), g.shift(-1), g.shift(1)
    ok = gap <= 14                                                       # 조사 재개일은 변화로 보지 않음
    if how in ("A", "R"):
        bad = ok & (d.abs() > np.log(1.5)) & ((nxt - prev).abs() < np.log(1.2))
    else:
        if fences is None:
            tr = ok & (d.abs() > 1e-9) & (k.date <= "2022-12-31")
            q1, q3 = np.percentile(d[tr], [25, 75])
            fences = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        bad = ok & (d.abs() > 1e-9) & ((d < fences[0]) | (d > fences[1]))
    return k.loc[bad, KM + ["date"]].assign(prev_price=np.exp(prev[bad]).values), fences


def build_variant(v):
    root = ROOT / "data" / f"processed_{v}"
    if all((root / d / "model_frame_test.parquet").exists() for d in NAMES):
        return root
    t0 = time.time()
    for sub in P0.iterdir():                                             # 다시 만들 파일 빼고는 원본을 하드링크로 공유 (복사 없음)
        if sub.is_dir():
            (root / sub.name).mkdir(parents=True, exist_ok=True)
            for f in sub.iterdir():
                if f.is_file() and f.name not in REBUILT and not f.name.startswith("model_frame") and not (root / sub.name / f.name).exists():
                    os.link(f, root / sub.name / f.name)
    for d in NAMES:
        p = pd.read_parquet(P0 / d / "price.parquet")
        ok = p.se_nm.isin(["소매", "중도매"]) & (p.price_kg > 0)
        bad, fences = flags(p[ok], v)
        m = p.merge(bad.assign(_bad=True), on=KM + ["date"], how="left")
        drop = m["_bad"].fillna(False).values.astype(bool)
        if v == "R":                                                     # 대치: 그날 가격 = 직전 조사일 가격
            p.loc[drop, "price_kg"] = m.loc[drop, "prev_price"].values
            p.to_parquet(root / d / "price.parquet", index=False)
        else:
            p[~drop].to_parquet(root / d / "price.parquet", index=False)
        print(f"[{v}] {NAMES[d]} 이상치 (시장·날짜) {len(bad):,} / 조사값 {'대치' if v == 'R' else '제외'} {drop.sum():,}행 ({drop.mean():.2%})"
              + (f", 울타리 {np.expm1(fences[0]):+.0%} ~ {np.expm1(fences[1]):+.0%}" if fences else ""), flush=True)
    bd.P = bfe.P = root                                                  # 데이터셋·피처를 걸러낸 가격으로 다시 만듦
    for domain, fn in bd.DOMAIN_VARS.items():
        df, daily = bd.base_rows(domain)
        df = fn(bd.common_vars(df), daily).sort_values(["ctgry_cd", "item_cd", "vrty_nm", "grd_nm", "se_nm", "date"]).reset_index(drop=True)
        df.insert(0, "domain", domain)
        df = df.loc[:, df.notna().any()].drop(columns=bd.SHAP_DROP[domain], errors="ignore")
        df.to_parquet(root / domain / "dataset.parquet", index=False)
    for d in NAMES:
        df = bfe.add_history_and_target(pd.read_parquet(root / d / "dataset.parquet"))
        df["woy"] = df.date.dt.isocalendar().week.astype("int16")
        cols = bfe.ID + bfe.COMMON + bfe.DOMAIN[d] + bfe.HIST + ["woy", "base", "y"]
        df.loc[df.y.notna(), [c for c in cols if c in df.columns]].reset_index(drop=True).to_parquet(root / d / "features.parquet", index=False)
    dv.cs.rm.P = dv.cs.P = dv.cs.fs.P = root                             # 학습 표 (최종 구성과 같은 함수)
    dv.cs.rm.SPLIT = dv.cs.SPLIT = dv.cs.fs.SPLIT = SPLIT
    for d in NAMES:
        df, num, cats = dv.final_frame(d)
        keep = sorted(set(dv.cs.KEYS + ["sgg_nm", "date", "y", "is_nat", "food_group", "dev_last", "base"] + num + cats))
        out = df[[c for c in keep if c in df.columns]].copy()
        for c, ok in df.attrs.get("scope", {}).items():
            out[f"_scope__{c}"] = ok
        for c in out.columns:
            if out[c].dtype == "float64":
                out[c] = out[c].astype("float32")
        out.attrs = {}
        out.to_parquet(root / d / "model_frame_test.parquet", index=False)
        print(f"[{v}] {NAMES[d]} 학습 표 {len(out):,}행", flush=True)
    dv.cs.rm.P = dv.cs.P = dv.cs.fs.P = P0
    print(f"[{v}] 재구성 {time.time() - t0:.0f}초", flush=True)
    return root


def load(root, d):
    df = pd.read_parquet(root / d / "model_frame_test.parquet")
    return df


def run():
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["분야"], r["데이터"], r["학습 처리"], r["구간"]) for r in rows}
    roots = {"원본": P0, "A 하루짜리 오류 제거": build_variant("A"), "B 1.5 IQR 밖 제거": build_variant("B"), "R 하루짜리 튐 → 전날 값": build_variant("R")}
    for d in ["livestock", "fishery", "agri"]:
        nm = NAMES[d]
        num = json.loads((P0 / d / "model_frame_test.json").read_text(encoding="utf-8"))["num"]
        cats = dv.base_sets(nm)[1]
        O = load(P0, d)
        mono = {} if d == "agri" else {k: v for k, v in dv.MONO[d].items() if k in num and k not in LEVEL}
        for vname, root in roots.items():
            V = O if root == P0 else load(root, d)
            for sname, (tend, es, ee) in SPLITS.items():
                todo = [L for L in ["없음", "1~99% 자르기", "Huber"] if (nm, vname, L, sname) not in done]
                if not todo:
                    continue
                df = pd.concat([V[(V.date >= "2016-01-01") & (V.date <= tend)], O[(O.date >= es) & (O.date <= ee) & O.y.notna()]], ignore_index=True)
                df.attrs["scope"] = {c[8:]: df[c].values.astype(bool) for c in df.columns if c.startswith("_scope__")}
                tr = (df.date <= tend).values & df.y.notna().values
                te = (df.date >= es).values
                X = dv.matrix(df, num, cats, tr, 42)
                yv = df.y.values.astype(float)
                ret = df.se_nm.values[te] == "소매"
                for L in todo:
                    t = time.time()
                    yt = np.clip(yv[tr], *np.nanquantile(yv[tr], (0.01, 0.99))) if L == "1~99% 자르기" else yv[tr]
                    pl = {**dv.LGB_BASE, "random_state": 42, **({"objective": "huber", "alpha": 0.05} if L == "Huber" else {})}
                    if mono:
                        pl.update(monotone_constraints=[mono.get(c, 0) for c in X.columns], monotone_constraints_method="advanced")
                    p = lgb.LGBMRegressor(**pl).fit(X[tr], yt, categorical_feature=cats).predict(X[te])
                    r = dv.metrics(yv[te][ret], p[ret])
                    rows.append({"분야": nm, "데이터": vname, "학습 처리": L, "구간": sname, "학습 행": int(tr.sum()),
                                 "RMSE": round(r["RMSE"], 5), "R²": round(r["R2"], 4), "방향 정확도": round(r["방향정확도"], 4), "초": round(time.time() - t)})
                    print(rows[-1], flush=True)
                    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
                del df, X
            if V is not O:
                del V
        del O
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(R.pivot_table(index=["분야", "데이터"], columns=["구간", "학습 처리"], values="R²").round(3).to_string())


if __name__ == "__main__":
    run()
