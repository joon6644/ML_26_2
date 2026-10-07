# %% [markdown]
# # 지역 + 전국 통합 모델에서 피처 선택 검증 (data/reference/feature_selection.csv)
# 통합 모델 = 14번과 같은 구성 (지역 행 + 전국 행, 지역 범주 sgg_nm, 전국 가격 이력 nat_*, 지역 프리미엄)
# 비교: ① 전체 피처 (14번 피처 전부 + 신규 3개: 설·추석까지 일수, 같은 계열 4주 변화율) ② CSV 최종 선택 (통합 피처 포함)
# 모델: LightGBM. 시드 0으로 비교하고 차이가 애매하면 시드 1, 2 추가. 학습 2016~2022 / 검증 2023~2024, 지역 행과 전국 행을 합쳐 평가
# 실행: `python eda/22_combined_selection.py` → eda/22_combined_selection.md

# %%
import importlib.util
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eda" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rm = _load("rm", "13_regional_model.py")
fs = _load("fs", "21_final_selection.py")
P, KEYS, HIST, SPLIT = rm.P, rm.KEYS, rm.HIST, rm.SPLIT
SIG = {"농산물": 0.0005, "축산물": 0.0002, "수산물": 0.0004}   # 통합 모델 RMSE 흔들림 σ (시드+피처 순서, 5회): eda/tables/seed_order_sigma.csv
NEW = ["days_to_seollal", "days_to_chuseok", "grp_r28"]


def build(d):
    reg = rm.regional_rows(d)
    nat, ext = rm.national_rows(d)
    df = reg.merge(nat, on=KEYS + ["date"], how="left")
    del reg
    df["premium"] = df.base / df.nat_base - 1
    f = fs.frame(d)   # 전국 행: features.parquet + 설·추석 일수 + grp_r28 + 통합 피처
    t0, t1, s0, s1 = SPLIT
    f = f[((f.date >= t0) & (f.date <= t1)) | ((f.date >= s0) & (f.date <= s1))].copy()
    # 지역 행에도 같은 시계열·날짜의 설·추석 일수와 grp_r28(전국 기준 같은 계열 흐름)을 붙임
    df = df.merge(f[KEYS + ["date"] + NEW], on=KEYS + ["date"], how="left")
    df = fs.add_events(df, d)   # 태풍·강풍 / 감염병 / 풍랑 감지 피처
    f["sgg_nm"] = "전국"
    for h in HIST:
        f[f"nat_{h}"] = f[h]
    f["premium"] = 0.0
    f["nat_base"] = f.base
    cols = [c for c in df.columns if c in f.columns]
    df = pd.concat([df[cols].assign(is_nat=0), f[cols].assign(is_nat=1)], ignore_index=True)
    df["woy"] = df.date.dt.isocalendar().week.astype("int16")
    tr = (df.date >= t0) & (df.date <= t1)
    for new, (ds, src, how) in fs.COMPOSITES.items():
        if d in ds:
            X = df[src] if how == "raw" else df[src] / df.loc[tr, src].mean()
            df[new] = X.mean(axis=1)
    df = fs.add_item_matched(df, d)   # 품목 대응 수입물가
    df.attrs["scope"] = fs.in_scope(df, d)   # 품목 대응 피처의 대상 품목 여부
    allnum = ["price_kg"] + HIST + ["woy"] + ext + [f"nat_{h}" for h in HIST] + ["premium"] + NEW
    return df, allnum


def fit(df, num, cats, tr, te, seed, scope=True):
    # 피처 순서도 시드로 섞음: colsample이 순서에 따라 다른 피처를 뽑아, 순서만 바꿔도 결과가 시드만큼 흔들림
    num = sorted(num)
    num = [num[i] for i in np.random.default_rng(seed).permutation(len(num))]
    Xc = rm.prep(df, num, tr)
    if scope:   # 대상 품목이 아닌 행은 -1 (MinMax 범위 밖이라 트리가 "해당 없음"으로 따로 나눔)
        for c, ok in df.attrs.get("scope", {}).items():
            if c in Xc:
                Xc.loc[~ok, c] = -1.0
    for c in cats:
        Xc[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    m = lgb.LGBMRegressor(**{**rm.LGB, "random_state": seed}).fit(Xc[tr], df.y.values[tr], categorical_feature=cats)
    return m.predict(Xc[te])


def main():
    t_all = time.time()
    sel = pd.read_csv(ROOT / "data" / "reference" / "feature_selection.csv")
    rows, notes = [], []
    for d, nm in rm.NAMES.items():
        t = time.time()
        df, allnum = build(d)
        t0, t1, s0, s1 = SPLIT
        tr = ((df.date >= t0) & (df.date <= t1)).values
        te = ((df.date >= s0) & (df.date <= s1)).values
        keep = sel[sel.분야 == nm].feature.tolist()
        cats = [c for c in fs.bf.CAT + ["sgg_nm"] if c in keep]
        num = [c for c in keep if c not in cats]   # CSV에 통합 모델 전용(nat_*, premium, sgg_nm)까지 포함
        miss = [c for c in num if c not in df.columns]
        assert not miss, miss
        sets = {"전체 피처": (allnum, fs.bf.CAT + ["sgg_nm"]), "최종 선택": (num, cats)}
        y, nat = df.y.values[te], df.is_nat.values[te] == 1
        parts = {"검증 전체 (지역+전국 행)": np.ones(len(y), bool)}   # 지역 행과 전국 행을 합쳐 한 번에 평가
        preds = {k: [fit(df, n, c, tr, te, 0)] for k, (n, c) in sets.items()}
        rmse = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
        diffs = [rmse(preds["최종 선택"][0], m) - rmse(preds["전체 피처"][0], m) for m in parts.values()]
        seeds = [0]
        if any(SIG[nm] <= abs(x) <= 3 * SIG[nm] for x in diffs):
            for s in (1, 2):
                for k, (n, c) in sets.items():
                    preds[k].append(fit(df, n, c, tr, te, s))
            seeds = [0, 1, 2]
        dev = np.expm1(df.dev_last.fillna(0).values[te])
        for part, m in parts.items():
            rows.append({"분야": nm, "평가 대상": part, "모델": "베이스라인 1 (오늘 가격 유지)", "피처 수": "-",
                         **{k: round(v, 5) for k, v in rm.metrics(y[m], dev[m]).items()}, "시드": "-"})
            per = {}
            for k, (n, c) in sets.items():
                ms = [rm.metrics(y[m], p[m]) for p in preds[k]]
                per[k] = [x["RMSE"] for x in ms]
                rows.append({"분야": nm, "평가 대상": part, "모델": f"LightGBM {k}", "피처 수": len(n) + len(c),
                             **{a: round(np.mean([x[a] for x in ms]), 5) for a in ms[0]}, "시드": ",".join(map(str, seeds))})
            diff = np.mean(per["최종 선택"]) - np.mean(per["전체 피처"])
            noise = SIG[nm] if len(seeds) == 1 else max(SIG[nm], 2 * np.std(np.subtract(per["최종 선택"], per["전체 피처"])))
            rows[-1]["ΔRMSE vs 전체"] = round(diff, 5)
            rows[-1]["판정"] = "유지 (잡음 이내)" if abs(diff) <= noise else ("나빠짐" if diff > 0 else "좋아짐")
        notes.append(f"- {nm}: 학습 {tr.sum():,}행 / 검증 {te.sum():,}행 (지역 {(~nat).sum():,}, 전국 {nat.sum():,}), 시드 {','.join(map(str, seeds))}, {time.time() - t:.0f}초")
        print(pd.DataFrame([r for r in rows if r["분야"] == nm]).drop(columns="분야").to_string(index=False), flush=True)
        del df
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "eda" / "tables" / "combined_selection.csv", encoding="utf-8-sig", index=False)
    L = ["# 지역 + 전국 통합 모델 피처 선택 검증 (검증 2023~2024)", "", "> 자동 생성: `python eda/22_combined_selection.py`",
         "> 전체 = 14번 피처 전부 + 신규 3개, 최종 = feature_selection.csv (통합 피처 포함) + 전국 가격 이력 + 지역 프리미엄 + 지역 범주",
         "> 시드 0으로 비교, 차이가 애매하면(σ~3σ) 시드 1, 2 추가", ""] + notes + [""]
    for nm in rm.NAMES.values():
        L += [f"## {nm}", "", R[R.분야 == nm].drop(columns="분야").fillna("").to_markdown(index=False, disable_numparse=True), ""]
    L.append(f"> 전체 소요 {time.time() - t_all:.0f}초")
    (ROOT / "eda" / "22_combined_selection.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
