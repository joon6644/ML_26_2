# %% [markdown]
# # 선형회귀 · XGBoost · LSTM 비교 (향후 30일 평균 가격)
# 학습 2016~2022, 테스트 2023~2024. 전처리(세 모델 공통): 결측 = 학습셋 평균, MinMax 스케일링(학습셋으로만 fit).
# 피처·타깃 정의는 03_baseline_models.py와 같다. LSTM은 GPU(PyTorch) 사용.
# 실행: `python eda/05_models_h30_2023_24.py` → eda/05_models_h30_2023_24.md, eda/tables/m30_*.csv

# %%
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.argv = [sys.argv[0], "30"]                                   # 03 모듈의 예측 기간 H = 30
spec = importlib.util.spec_from_file_location("bm", ROOT / "eda" / "03_baseline_models.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

import torch                                                     # noqa: E402
import xgboost as xgb                                            # noqa: E402
from sklearn.linear_model import LinearRegression                # noqa: E402
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder    # noqa: E402

H = 30
TRAIN_START, TRAIN_END = "2016-01-01", "2022-12-01"   # 타깃 창(t+1~t+30)이 2022-12-31 안에 끝남
VAL_START = "2022-06-01"                              # LSTM 조기 종료용 (학습 기간 안에서 떼어냄)
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
SEQ = 30                                              # LSTM 입력: 최근 30개 조사일
DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = ROOT / "data" / "processed"
TAB = ROOT / "eda" / "tables"
torch.manual_seed(0)
np.random.seed(0)


def load_domain(domain, common, item_map):
    daily, keys = bm.build_series(domain)
    df = bm.make_rows(daily, keys).merge(common, on="date", how="left").merge(item_map, on=["ctgry_cd", "item_cd"], how="left")
    fn, dom_cols = bm.DOMAIN_FEATURES[domain]
    df = fn(df).replace([np.inf, -np.inf], np.nan)
    df["sid"] = df.groupby(keys, observed=True).ngroup()
    df = df.sort_values(["sid", "date"]).reset_index(drop=True)          # LSTM 창을 위해 시계열·날짜 순 정렬
    return df, bm.NUM + dom_cols


def preprocess(df, num):
    """결측 = 학습셋 평균, MinMax = 학습셋 fit. 반환: 전체 행의 스케일된 수치 행렬."""
    tr = (df.date >= TRAIN_START) & (df.date <= TRAIN_END)
    means = df.loc[tr, num].mean()
    X = df[num].fillna(means)
    sc = MinMaxScaler().fit(X[tr])
    return pd.DataFrame(sc.transform(X), columns=num, index=df.index).astype("float32")


# %% [markdown]
# ## LSTM

# %%
class LSTMReg(torch.nn.Module):
    def __init__(self, n_feat, n_item, emb=16, hidden=64):
        super().__init__()
        self.emb = torch.nn.Embedding(n_item, emb)
        self.lstm = torch.nn.LSTM(n_feat + emb, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x, item):
        e = self.emb(item).unsqueeze(1).expand(-1, x.shape[1], -1)
        out, _ = self.lstm(torch.cat([x, e], dim=2))
        return self.head(out[:, -1]).squeeze(1)


def windows(idx, start):
    """각 행 idx에 대해 같은 시계열의 직전 SEQ개 행 인덱스 (시계열 시작 전은 첫 행으로 채움)."""
    off = torch.arange(-SEQ + 1, 1, device=idx.device)
    w = idx[:, None] + off[None, :]
    return torch.maximum(w, start[idx][:, None])


def train_lstm(df, Xs, y, tr_idx, va_idx, te_idx, item_idx, epochs=30, bs=2048, patience=4):
    X = torch.tensor(Xs.values, device=DEV)
    sid = df.sid.values
    first = pd.Series(np.arange(len(df))).groupby(sid).transform("min").values      # 각 행이 속한 시계열의 첫 행
    start = torch.tensor(first, device=DEV)
    items = torch.tensor(item_idx, device=DEV)
    y_mu, y_sd = y[tr_idx].mean(), y[tr_idx].std()
    Y = torch.tensor(((y - y_mu) / y_sd).astype("float32"), device=DEV)              # 타깃 표준화 (학습셋 기준)
    model = LSTMReg(X.shape[1], int(item_idx.max()) + 1).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()

    def predict(rows):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 8192):
                r = torch.tensor(rows[i:i + 8192], device=DEV)
                out.append(model(X[windows(r, start)], items[r]))
        return torch.cat(out).cpu().numpy() * y_sd + y_mu

    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), bs):
            r = torch.tensor(perm[i:i + bs], device=DEV)
            opt.zero_grad()
            loss = lossf(model(X[windows(r, start)], items[r]), Y[r])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va = np.sqrt(np.mean((predict(va_idx) - y[va_idx]) ** 2))
        hist.append(va)
        if va < best - 1e-5:
            best, best_state, wait = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return predict(te_idx), len(hist), hist


# %% [markdown]
# ## 평가

# %%
def metrics(y, yhat, base):
    fut, ph = base * np.exp(y), base * np.exp(yhat)
    big = np.abs(y) >= 0.10
    moved = np.abs(y) >= 0.01
    return {"RMSE": np.sqrt(np.mean((yhat - y) ** 2)), "MAE": np.mean(np.abs(yhat - y)), "R2": bm.r2(y, yhat),
            "MAPE(%)": np.mean(np.abs(ph - fut) / fut) * 100,
            "방향정확도(%)": np.mean(np.sign(yhat[moved]) == np.sign(y[moved])) * 100 if np.any(yhat != 0) else np.nan,
            "급변 MAPE(%)": np.mean(np.abs(ph[big] - fut[big]) / fut[big]) * 100,
            "급변 방향(%)": np.mean(np.sign(yhat[big]) == np.sign(y[big])) * 100 if np.any(yhat != 0) else np.nan,
            "급변 크기비": np.median(np.abs(yhat[big]) / np.abs(y[big]))}


def run(domain, common, item_map):
    t0 = time.time()
    df, num = load_domain(domain, common, item_map)
    ok = df.y.notna() & df.base.notna()
    tr_m = ok & (df.date >= TRAIN_START) & (df.date <= TRAIN_END)
    te_m = ok & (df.date >= TEST_START) & (df.date <= TEST_END)
    Xs = preprocess(df, num)
    y = df.y.values.astype("float32")
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    preds, secs = {}, {}

    preds["기준: 오늘 가격 유지"] = df.dev_last.fillna(0).values[te]
    preds["기준: 작년 같은 시기"] = df.ly_change.fillna(0).values[te]

    t = time.time()
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(df.loc[tr, bm.LIN_CAT].astype(str))
    def lin_x(rows):
        from scipy.sparse import hstack, csr_matrix
        return hstack([csr_matrix(Xs.values[rows]), oh.transform(df.loc[rows, bm.LIN_CAT].astype(str))]).tocsr()
    lin = LinearRegression().fit(lin_x(tr), y[tr])
    preds["선형회귀"] = lin.predict(lin_x(te))
    secs["선형회귀"] = time.time() - t

    t = time.time()
    Xx = Xs.copy()
    for c in bm.CAT:
        Xx[c] = df[c].astype(str).astype(pd.CategoricalDtype(sorted(df[c].astype(str).unique())))
    xg = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                          tree_method="hist", device=DEV, enable_categorical=True, random_state=0)
    xg.fit(Xx.iloc[tr], y[tr])
    preds["XGBoost"] = xg.predict(Xx.iloc[te])
    secs["XGBoost"] = time.time() - t

    t = time.time()
    va = np.where(tr_m & (df.date >= VAL_START))[0]
    tr_l = np.where(tr_m & (df.date < pd.Timestamp(VAL_START) - pd.Timedelta(days=H)))[0]   # 검증 타깃 창과 겹치지 않게
    item_idx = df.item_cd.astype("category").cat.codes.values.astype("int64")
    lstm_pred, n_ep, hist = train_lstm(df, Xs, y, tr_l, va, te, item_idx)
    preds["LSTM"] = lstm_pred
    secs["LSTM"] = time.time() - t

    base = df.base.values[te]
    res = pd.DataFrame({k: metrics(y[te], np.asarray(v, dtype="float64"), base) for k, v in preds.items()}).T
    res["오늘가격유지 대비 RMSE 개선(%)"] = (1 - res.RMSE / res.loc["기준: 오늘 가격 유지", "RMSE"]) * 100
    res["학습시간(초)"] = pd.Series(secs)
    out = P / "baseline_preds"
    out.mkdir(exist_ok=True)
    df.loc[te, ["date", "item_nm", "vrty_nm", "grd_nm", "se_nm", "base", "y"]].assign(
        **{f"pred_{k}": np.asarray(v) for k, v in preds.items()}).to_parquet(out / f"m30_{domain}.parquet", index=False)
    info = {"시계열 수": int(df.sid.nunique()), "학습 행": len(tr), "LSTM 학습/검증 행": f"{len(tr_l):,} / {len(va):,}", "테스트 행": len(te),
            "피처 수": len(num), "LSTM 에폭": n_ep, "검증 RMSE 추이": " → ".join(f"{h:.4f}" for h in hist), "전체 소요(초)": round(time.time() - t0)}
    print(domain, info)
    return res, info


# %%
if __name__ == "__main__":
    common = bm.common_features()
    item_map = pd.read_csv(ROOT / "data" / "reference" / "item_map.csv", dtype=str)[["ctgry_cd", "item_cd", "food_group"]]
    names = {"agri": "농산물", "livestock": "축산물", "fishery": "수산물"}
    lines = ["# 선형회귀 · XGBoost · LSTM (향후 30일 평균 가격)", "",
             "> 자동 생성: `python eda/05_models_h30_2023_24.py`",
             f"> 타깃 y = log(향후 {H}일 평균 / 최근 7일 평균). 학습 origin {TRAIN_START} ~ {TRAIN_END}, **테스트 origin {TEST_START} ~ {TEST_END}**",
             "> 전처리(세 모델 공통): 결측 = 학습셋 평균, MinMax 스케일링(학습셋 fit). LSTM: 최근 30개 조사일 시퀀스 + 품목 임베딩, 2층(hidden 64), "
             f"조기 종료용 검증 = 학습 기간 중 {VAL_START} 이후, 장치 = {DEV} ({torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU'})",
             "> RMSE·MAE·R²는 변화율(log) 스케일. 급변 = 실제 |변화| ≥ 10%. 급변 크기비 = 예측 변화 / 실제 변화 (중앙값, 1이면 크기까지 맞춤)", ""]
    allres = []
    for d in ("agri", "livestock", "fishery"):
        res, info = run(d, common, item_map)
        res.to_csv(TAB / f"m30_{d}.csv", encoding="utf-8-sig")
        allres.append(res.assign(도메인=names[d]))
        lines += [f"## {names[d]}", "", " · ".join(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}" for k, v in info.items()), "",
                  res.round(4).reset_index(names="모델").to_markdown(index=False, disable_numparse=True), ""]
    s = pd.concat(allres).reset_index(names="모델")
    s.to_csv(TAB / "m30_summary.csv", encoding="utf-8-sig", index=False)
    lines[7:7] = ["## 요약", "", s[["도메인", "모델", "RMSE", "MAE", "R2", "MAPE(%)", "방향정확도(%)", "급변 방향(%)", "급변 크기비", "학습시간(초)"]]
                  .round(4).to_markdown(index=False, disable_numparse=True), ""]
    (ROOT / "eda" / "05_models_h30_2023_24.md").write_text("\n".join(lines), encoding="utf-8")
    print(s[["도메인", "모델", "RMSE", "MAE", "R2", "MAPE(%)", "방향정확도(%)", "급변 크기비", "학습시간(초)"]].round(4).to_string(index=False))
