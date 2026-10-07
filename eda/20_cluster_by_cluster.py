# %% [markdown]
# # 고상관 피처 묶음별 압축 실험 (묶음 하나씩, 대표 후보 전부 비교)
# 묶음: 학습 기간 Spearman |ρ| ≥ 0.9 / ≥ 0.8 complete-linkage 군집 (중복 제거)
# 각 묶음마다: (a) 구성원 하나씩 남기고 나머지 제거 (구성원 수만큼), (b) 묶음 전체 제거. 다른 피처는 모두 유지
# 비교: 전체 피처 모델과 같은 시드 3개 평균. 판정 |ΔRMSE| ≤ 2·√(σ전체² + σ실험²) → 유지
# 모델: LightGBM, 전국(features.parquet), 학습 2016~2022 / 테스트 2023~2024
# 실행: `python eda/20_cluster_by_cluster.py` → eda/20_cluster_by_cluster.md, eda/tables/cluster_by_cluster.csv

# %%
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eda" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fd = load("fd", "17_feature_decision.py")
rr = load("rr", "18_redundancy_reduce.py")
SEEDS = (0, 1, 2)


def runs(f, num, tr, te):
    r = [fd.fit(f, num, tr, te, seed=s) for s in SEEDS]
    return np.mean([x["RMSE"] for x in r]), np.std([x["RMSE"] for x in r]), np.mean([x["R2"] for x in r])


def main():
    t_all = time.time()
    rows = []
    L = ["# 고상관 피처 묶음별 압축 실험", "", "> 자동 생성: `python eda/20_cluster_by_cluster.py`",
         "> 묶음 하나씩: 구성원 하나만 남기기(후보별) / 묶음 전체 제거. 다른 피처는 모두 유지. LightGBM 시드 3개 평균, 테스트 2023~2024",
         "> ΔRMSE = 실험 − 전체 (양수 = 나빠짐). 판정: |ΔRMSE| ≤ 2·√(σ전체² + σ실험²) 이면 유지", ""]
    for d, nm in fd.NAMES.items():
        t = time.time()
        f = pd.read_parquet(fd.P / d / "features.parquet")
        t0, t1, s0, s1 = fd.SPLIT
        tr = ((f.date >= t0) & (f.date <= t1)).values
        te = ((f.date >= s0) & (f.date <= s1)).values
        num = [c for c in f.columns if c not in fd.IDS]
        shap = pd.read_csv(fd.TAB / f"feature_decision_{d}.csv").set_index("피처")["SHAP %"]
        corr = f[tr].sample(min(200000, tr.sum()), random_state=0)[num].corr(method="spearman")
        cls = []
        for thr in (0.9, 0.8):
            for g in rr.clusters(corr, thr):
                if sorted(g) not in [sorted(x[1]) for x in cls]:
                    cls.append((thr, g))
        full = runs(f, num, tr, te)
        L += [f"## {nm}", "", f"- 전체 피처 {len(num) + 3}개: RMSE {full[0]:.5f} ± {full[1]:.5f}, R² {full[2]:.4f}", f"- 묶음 {len(cls)}개", ""]
        for k, (thr, g) in enumerate(cls, 1):
            minr = corr.loc[g, g].abs().values[np.triu_indices(len(g), 1)].min()
            opts = [(f"{c} 만 남김", [x for x in g if x != c], c) for c in sorted(g, key=lambda c: -shap.get(c, 0))]
            opts.append(("묶음 전체 제거", g, ""))
            sub = []
            for label, drop, keep in opts:
                r = runs(f, [c for c in num if c not in drop], tr, te)
                diff, noise = r[0] - full[0], np.hypot(full[1], r[1])
                verdict = "유지" if abs(diff) <= 2 * noise else ("나빠짐" if diff > 0 else "좋아짐")
                row = {"분야": nm, "묶음": k, "임계값": thr, "묶음 내 최소 |ρ|": round(minr, 2), "구성원": ", ".join(g),
                       "실험": label, "남긴 피처 SHAP %": round(shap.get(keep, np.nan), 2) if keep else "",
                       "RMSE": round(r[0], 5), "±": round(r[1], 5), "R²": round(r[2], 4), "ΔRMSE": round(diff, 5),
                       "ΔR²": round(r[2] - full[2], 4), "판정": verdict}
                rows.append(row)
                sub.append(row)
            print(f"  {nm} 묶음 {k}/{len(cls)} ({len(g)}개) 완료", flush=True)
            best = min([s for s in sub if s["실험"] != "묶음 전체 제거"], key=lambda s: s["ΔRMSE"])
            L += [f"### 묶음 {k} (|ρ| ≥ {thr}, 묶음 내 최소 {minr:.2f}): {', '.join(g)}", "",
                  f"가장 좋은 대표: **{best['실험'].replace(' 만 남김', '')}** (ΔRMSE {best['ΔRMSE']:+.5f}, {best['판정']})", "",
                  pd.DataFrame(sub)[["실험", "남긴 피처 SHAP %", "RMSE", "±", "R²", "ΔRMSE", "ΔR²", "판정"]].to_markdown(index=False), ""]
        print(f"{nm} 완료 {time.time() - t:.0f}초", flush=True)
        pd.DataFrame(rows).to_csv(fd.TAB / "cluster_by_cluster.csv", encoding="utf-8-sig", index=False)
        (ROOT / "eda" / "20_cluster_by_cluster.md").write_text("\n".join(L), encoding="utf-8")
    L.append(f"> 전체 소요 {time.time() - t_all:.0f}초")
    (ROOT / "eda" / "20_cluster_by_cluster.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
