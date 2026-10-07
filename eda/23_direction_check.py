# %% [markdown]
# # 피처 방향 점검: 사전 가설(feature_selection.csv '기대 방향') vs SHAP
# 통합 모델(22번 구성, 최종 선택, 시드 0)에서 검증 행 표본의 SHAP 값을 뽑아, 피처 값이 클수록 예측을 올렸는지 내렸는지 확인
# 방향 ρ = 품목별로 평균을 뺀 피처 값과 SHAP 값의 Spearman 상관 (품목 간 수준 차이를 제거하고 품목 안에서의 방향만 봄)
# 대상 품목이 정해진 피처는 대상 품목 행만 사용. 판정: |ρ| < 0.1 방향 없음 / 부호 일치 / 반대. 기대 방향이 '조건부'면 방향 판정 제외
# 추세 ρ = 학습기간 날짜별 값과 날짜의 Spearman 상관. |추세 ρ| ≥ 0.8이면 시간 흐름 변수로 쓰였을 가능성 표시
# 실행: `python eda/23_direction_check.py` → eda/23_direction_check.md

# %%
import importlib.util
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cs", ROOT / "eda" / "22_combined_selection.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
N = 40000


def main():
    sel = pd.read_csv(ROOT / "data" / "reference" / "feature_selection.csv")
    rows = []
    for d, nm in cs.rm.NAMES.items():
        df, _ = cs.build(d)
        t0, t1, s0, s1 = cs.SPLIT
        tr = ((df.date >= t0) & (df.date <= t1)).values
        te = ((df.date >= s0) & (df.date <= s1)).values
        g = sel[sel.분야 == nm]
        cats = [c for c in cs.fs.bf.CAT + ["sgg_nm"] if c in g.feature.values]
        num = sorted(c for c in g.feature if c not in cats)
        Xc = cs.rm.prep(df, num, tr)
        scope = df.attrs["scope"]
        for c, ok in scope.items():
            if c in Xc:
                Xc.loc[~ok, c] = -1.0
        for c in cats:
            Xc[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
        m = lgb.LGBMRegressor(**{**cs.rm.LGB, "random_state": 0}).fit(Xc[tr], df.y.values[tr], categorical_feature=cats)
        idx = np.random.default_rng(0).choice(np.where(te)[0], min(N, te.sum()), replace=False)
        sh = pd.DataFrame(m.predict(Xc.iloc[idx], pred_contrib=True)[:, :-1], columns=Xc.columns)
        tot = sh.abs().mean()
        tot = tot / tot.sum() * 100
        item = df.item_cd.values[idx]
        daily = df.loc[tr].groupby("date")[num].mean()
        for r in g.itertuples():
            f = r.feature
            exp = g.loc[g.feature == f, "기대 방향"].iloc[0]
            row = {"분야": nm, "feature": f, "한글명": r.한글명, "기대 방향": exp, "SHAP %": round(tot[f], 2)}
            if exp != "범주":
                ok = scope.get(f, np.ones(len(df), bool))[idx]
                x = pd.Series(Xc[f].values[idx][ok]).astype(float)
                s = pd.Series(sh[f].values[ok])
                it = pd.Series(item[ok])
                xw, sw = x - x.groupby(it.values).transform("mean"), s - s.groupby(it.values).transform("mean")
                rho = xw.corr(sw, method="spearman") if xw.std() > 0 else np.nan
                trend = daily[f].corr(pd.Series(daily.index.map(pd.Timestamp.toordinal), index=daily.index), method="spearman")
                row.update({"방향 ρ": round(rho, 2), "대상 행 %": round(ok.mean() * 100), "추세 ρ": round(trend, 2)})
                if exp == "조건부":
                    row["판정"] = "조건부 (방향 판정 제외)"
                elif not np.isfinite(rho) or abs(rho) < 0.1:
                    row["판정"] = "방향 없음"
                else:
                    row["판정"] = "일치" if (rho > 0) == (exp == "+") else "반대"
                if abs(trend) >= 0.8:
                    row["비고"] = "추세형 (시간 변수 의심)"
            rows.append(row)
        print(nm, "done", flush=True)
        del df, Xc
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "eda" / "tables" / "direction_check.csv", encoding="utf-8-sig", index=False)
    L = ["# 피처 방향 점검 (사전 가설 vs SHAP, 통합 모델 · 검증 2023~2024)", "", "> 자동 생성: `python eda/23_direction_check.py`",
         "> 방향 ρ = 품목 내 피처 값과 SHAP 값의 Spearman 상관. |ρ| < 0.1 방향 없음. 추세 ρ ≥ 0.8은 시간 흐름 변수 의심", ""]
    for nm in cs.rm.NAMES.values():
        L += [f"## {nm}", "", R[R.분야 == nm].drop(columns="분야").fillna("").to_markdown(index=False, disable_numparse=True), ""]
    (ROOT / "eda" / "23_direction_check.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
