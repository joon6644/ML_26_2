# %% [markdown]
# # 피처 다이어트 (SHAP 순위로 원본 변수 줄이기)
# 선정: 2016~2021 학습 → 2022 검증셋에서 SHAP 비중(LightGBM·XGBoost 평균)으로 원본 변수 순위. 테스트(2023~24)는 선정에 쓰지 않음
# 가격 이력 8개·주차·범주형(item_cd·food_group·se_nm)은 항상 유지. 원본 변수를 ① SHAP 상위 50% ② 도메인 기준 정리(curated)로 줄여 비교
# 평가: 확정 포맷 (학습 2016~2022 / 테스트 2023~2024, 베이스라인 vs 모델)
# 실행: `python eda/10_feature_diet.py` → eda/10_feature_diet.md, eda/tables/feature_diet_rank.csv

# %%
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eda" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev = load("ev", "08_raw_dataset_eval.py")
sh = load("sh", "09_shap.py")
VAL = ("2016-01-01", "2021-12-01", "2022-01-01", "2022-12-03")   # 선정용: 검증 타깃이 2022년 안에서 끝나도록
KEEP = [1.0, 0.5]

# 도메인 기준 정리: 경제 지표는 해당 분야와 직접 관련된 것만 남긴다 (월별 지표는 모든 품목에 같은 값이라 노상관이어도 '시기 표식'으로 SHAP이 높게 나옴)
INTL_KEEP = {"agri": ["원유_Brent", "천연가스", "소맥", "옥수수", "대두"],
             "livestock": ["원유_Brent", "천연가스", "소맥", "옥수수", "대두"],      # 곡물 = 사료 원료
             "fishery": ["원유_Brent", "천연가스"]}                                 # 원유·가스 = 어선·양식장 연료
IMPIDX_KEEP = {"agri": ["농산물", "곡류", "곡물및식량작물", "맥류및잡곡", "콩류", "과일", "채소및과실", "과실및채소가공품", "기타식용작물", "유지"],
               "livestock": ["축산물", "도축육", "가금육", "낙농품", "낙농및육류", "육가공품및낙농품"],
               "fishery": ["수산물", "신선수산물", "냉동수산물", "냉동건조수산물", "수산가공품", "수산물가공품"]}


def curated(raw, d):
    """도메인 기준으로 노상관 변수 제거: 무관한 국제원자재·수입물가, 영양성분(품목 고정값 = item_cd와 중복), 월(주차와 중복)."""
    def ok(c):
        if c.startswith("intl_"):
            return c[5:] in INTL_KEEP[d]
        if c.startswith("impidx_"):
            return c[7:] in IMPIDX_KEEP[d]
        return not c.startswith("nutr_") and c != "month"
    return [c for c in raw if ok(c)]


def main():
    rows, ranks = [], []
    for d, nm in ev.NAMES.items():
        df = pd.read_parquet(ev.P / d / "dataset.parquet")
        df = ev.add_calendar(ev.add_base_and_history(df)).reset_index(drop=True)
        raw = [c for c in df.columns if c not in ev.DROP and c not in ev.CAT and c not in ev.CAL + ev.HIST + ["base", "y"]
               and pd.api.types.is_numeric_dtype(df[c])]
        res = sh.shap_run(df, raw + ev.CAL + ev.HIST, VAL)
        share = sum(s / s.sum() for s in res.values()) / len(res)
        rank = share[raw].sort_values(ascending=False)
        ranks.append(pd.DataFrame({"분야": nm, "피처": rank.index, "검증 SHAP 비중": rank.values, "순위": range(1, len(rank) + 1)}))
        for k in KEEP:
            keep = list(rank.index[: max(1, round(len(raw) * k))])
            out, _, ntr, nte = ev.fit_eval(df, keep + ev.CAL + ev.HIST, ev.SPLIT)
            for model, mt in out.items():
                rows.append({"분야": nm, "원본 변수": f"{len(keep)}개 ({k:.0%})", "모델": model, **mt})
            print(f"{nm} 원본 변수 {len(keep)}/{len(raw)} 완료", flush=True)
        keep = curated(raw, d)
        out, _, _, _ = ev.fit_eval(df, keep + ev.CAL + ev.HIST, ev.SPLIT)
        for model, mt in out.items():
            rows.append({"분야": nm, "원본 변수": f"도메인 정리 {len(keep)}개", "모델": model, **mt})
        print(f"{nm} 도메인 정리 {len(keep)}/{len(raw)} 완료", flush=True)
    r, rk = pd.DataFrame(rows), pd.concat(ranks)
    rk.to_csv(ev.TAB / "feature_diet_rank.csv", encoding="utf-8-sig", index=False)
    r.to_csv(ev.TAB / "feature_diet_eval.csv", encoding="utf-8-sig", index=False)
    lines = ["# 피처 다이어트 (테스트 2023~2024)", "", "> 자동 생성: `python eda/10_feature_diet.py`",
             "> 원본 변수를 2022 검증셋 SHAP 순위로 상위 K%만 남김. 가격 이력 8개·주차·범주형 3개는 항상 포함", ""]
    for nm in ev.NAMES.values():
        x = r[r.분야 == nm].drop(columns="분야")
        lines += [f"## {nm}", "", x.round(4).to_markdown(index=False, disable_numparse=True), ""]
    (ROOT / "eda" / "10_feature_diet.md").write_text(chr(10).join(lines), encoding="utf-8")
    print(r.pivot_table(index=["분야", "모델"], columns="원본 변수", values="R2", sort=False).round(4).to_string())


if __name__ == "__main__":
    main()
