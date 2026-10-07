# %% [markdown]
# # 상관 높은 피처 묶음에서 대표 1개만 남겨도 성능이 유지되는가 (전국 모델, LightGBM)
# 묶음: 학습 기간 Spearman |ρ| ≥ 임계값으로 계층 군집 (complete linkage → 묶음 안 모든 쌍이 임계값 이상)
# 대표: 묶음 안에서 SHAP 비중(eda/17, 테스트 2023~24)이 가장 큰 피처. 나머지는 모두 동시에 제거
# 비교: 전체 피처 vs 축소 피처, 각각 시드 3개 평균 ± 표준편차 (학습 2016~2022 / 테스트 2023~2024)
# 실행: `python eda/18_redundancy_reduce.py` → eda/18_redundancy_reduce.md

# %%
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("fd", ROOT / "eda" / "17_feature_decision.py")
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)
THRESHOLDS = [0.9, 0.8]
SEEDS = (0, 1, 2)


def clusters(corr, thr):
    d = 1 - corr.abs().fillna(0).values
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(squareform((d + d.T) / 2, checks=False), "complete"), t=1 - thr, criterion="distance")
    s = pd.Series(lab, index=corr.index)
    return [list(g.index) for _, g in s.groupby(s) if len(g) > 1]


def runs(f, num, tr, te):
    r = [fd.fit(f, num, tr, te, seed=s) for s in SEEDS]
    return np.mean([x["RMSE"] for x in r]), np.std([x["RMSE"] for x in r]), np.mean([x["R2"] for x in r])


def main():
    t_all = time.time()
    L = ["# 상관 높은 피처 묶음 → 대표 1개만 남기기", "", "> 자동 생성: `python eda/18_redundancy_reduce.py`",
         "> 묶음 = 학습 기간 Spearman |ρ| ≥ 임계값 (묶음 안 모든 쌍). 대표 = 묶음 안 SHAP 비중 최대. LightGBM 시드 3개 평균, 테스트 2023~2024", ""]
    summary = []
    for d, nm in fd.NAMES.items():
        f = pd.read_parquet(fd.P / d / "features.parquet")
        t0, t1, s0, s1 = fd.SPLIT
        tr = ((f.date >= t0) & (f.date <= t1)).values
        te = ((f.date >= s0) & (f.date <= s1)).values
        num = [c for c in f.columns if c not in fd.IDS]
        shap = pd.read_csv(fd.TAB / f"feature_decision_{d}.csv").set_index("피처")["SHAP %"]
        corr = f[tr].sample(min(200000, tr.sum()), random_state=0)[num].corr(method="spearman")
        full = runs(f, num, tr, te)
        L += [f"## {nm}", "", f"- 전체 피처 {len(num) + 3}개: RMSE {full[0]:.4f} ± {full[1]:.5f}, R² {full[2]:.3f}", ""]
        summary.append({"분야": nm, "피처 세트": f"전체 {len(num) + 3}개", "RMSE": round(full[0], 4), "± (시드)": round(full[1], 5), "R²": round(full[2], 3)})
        for thr in THRESHOLDS:
            cl = clusters(corr, thr)
            drop, rows = [], []
            for g in cl:
                rep = max(g, key=lambda c: shap.get(c, 0))
                drop += [c for c in g if c != rep]
                rows.append({"대표 (남김)": f"{rep} ({shap.get(rep, 0):.2f}%)",
                             "제거": ", ".join(f"{c} ({shap.get(c, 0):.2f}%)" for c in g if c != rep),
                             "묶음 내 최소 |ρ|": round(corr.loc[g, g].abs().values[np.triu_indices(len(g), 1)].min(), 2)})
            red = runs(f, [c for c in num if c not in drop], tr, te)
            diff = red[0] - full[0]
            noise = np.hypot(full[1], red[1])
            verdict = "유지 (차이가 시드 잡음 이내)" if abs(diff) <= 2 * noise else ("나빠짐" if diff > 0 else "좋아짐")
            summary.append({"분야": nm, "피처 세트": f"|ρ|≥{thr} 축소 {len(num) + 3 - len(drop)}개 (−{len(drop)})",
                            "RMSE": round(red[0], 4), "± (시드)": round(red[1], 5), "R²": round(red[2], 3),
                            "ΔRMSE": round(diff, 5), "판정": verdict})
            L += [f"### |ρ| ≥ {thr}: 묶음 {len(cl)}개, {len(drop)}개 제거 → 피처 {len(num) + 3 - len(drop)}개", "",
                  f"- 축소 모델: RMSE {red[0]:.4f} ± {red[1]:.5f}, R² {red[2]:.3f} → ΔRMSE {diff:+.5f} (판정 기준 ±{2 * noise:.5f}) → **{verdict}**", "",
                  pd.DataFrame(rows).to_markdown(index=False), ""]
            print(f"{nm} |ρ|≥{thr}: −{len(drop)}개, ΔRMSE {diff:+.5f} → {verdict}", flush=True)
    S = pd.DataFrame(summary)
    L = L[:4] + ["## 요약", "", S.fillna("").to_markdown(index=False), ""] + L[4:] + [f"> 전체 소요 {time.time() - t_all:.0f}초"]
    (ROOT / "eda" / "18_redundancy_reduce.md").write_text("\n".join(L), encoding="utf-8")
    S.to_csv(fd.TAB / "redundancy_reduce.csv", encoding="utf-8-sig", index=False)


if __name__ == "__main__":
    main()
