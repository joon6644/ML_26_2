# %% [markdown]
# # LSTM (지역 + 전국 통합 데이터, 향후 28일 평균 가격 변화율)
# 입력 = ① 과거 91일 가격 시퀀스 (log(가격 / 최근 7일 평균), 조사 여부 마스크) → LSTM
#        ② 통합 트리 모델과 같은 피처 (결측 = 학습셋 평균, MinMax) → MLP
#        ③ 품목·계열·소매/중도매·지역 임베딩
# 학습 2016-01 ~ 2022-05, 조기 종료 검증 2022-06 ~ 2022-12, 테스트 2023~2024 (지역 행 / 전국 행 따로 평가)
# 실행: `python eda/15_lstm_combined.py` → eda/15_lstm_combined.md

# %%
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cm", ROOT / "eda" / "14_combined_model.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
rm = cm.rm
P, KEYS, SPLIT, CAT = rm.P, rm.KEYS, rm.SPLIT, rm.CAT
RKEYS = KEYS + ["sgg_nm"]
L = 91
VAL_START = "2022-06-01"
DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)


def series_arrays(d):
    """시계열별 일 단위 log 가격 배열 (앞에 L일 NaN 패딩) → 하나로 이어 붙인 배열과 시계열별 (시작 위치, 시작일)."""
    p = pd.read_parquet(P / d / "price.parquet", columns=["date", "se_nm", "ctgry_cd", "item_cd", "item_nm", "vrty_nm", "grd_nm", "sgg_nm", "price_kg"])
    p = p[p.se_nm.isin(["소매", "중도매"]) & (p.price_kg > 0) & (p.sgg_nm != "전국") & (p.date >= "2015-01-01")]
    reg = p.groupby(RKEYS + ["date"], observed=True).price_kg.median().reset_index()
    ds = pd.read_parquet(P / d / "dataset.parquet", columns=KEYS + ["date", "price_kg"]).assign(sgg_nm="전국")
    allp = pd.concat([reg, ds], ignore_index=True)
    chunks, index, pos = [], {}, 0
    for k, g in allp.groupby(RKEYS, observed=True, sort=False):
        s = np.log(g.set_index("date").price_kg.asfreq("D")).astype("float32")
        arr = np.concatenate([np.full(L, np.nan, "float32"), s.values])
        index[k] = (pos + L, s.index[0])
        chunks.append(arr)
        pos += len(arr)
    return np.concatenate(chunks), index


class Net(nn.Module):
    def __init__(self, n_num, cards):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c + 1, min(16, (c + 1) // 2 + 1)) for c in cards])
        e = sum(m.embedding_dim for m in self.emb)
        self.lstm = nn.LSTM(2, 64, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(64 + n_num + e, 256), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, seq, num, cat):
        _, (h, _) = self.lstm(seq)
        z = torch.cat([h[-1], num] + [m(cat[:, i]) for i, m in enumerate(self.emb)], 1)
        return self.mlp(z).squeeze(1)


def main():
    t_all = time.time()
    rows = []
    only = os.environ.get("DOMAINS")
    for d, nm in rm.NAMES.items():
        if only and d not in only.split(","):
            continue
        t = time.time()
        df, num = cm.build(d)
        flat, index = series_arrays(d)
        keyt = list(zip(*[df[c] for c in RKEYS]))
        starts = np.array([index[k][0] for k in keyt], dtype="int64")
        d0 = pd.to_datetime([index[k][1] for k in keyt])
        df["pos"] = starts + (df.date.values - d0.values).astype("timedelta64[D]").astype("int64")
        t0, t1, s0, s1 = SPLIT
        tr_all = ((df.date >= t0) & (df.date <= t1)).values
        tr_m = tr_all & (df.date < VAL_START).values
        va_m = tr_all & (df.date >= VAL_START).values
        te_m = ((df.date >= s0) & (df.date <= s1)).values
        Xs = np.clip(rm.prep(df, num, tr_all).values, 0, 1)   # 학습 범위 밖 값은 경계로 (신경망의 선형 외삽 방지)
        cats = np.stack([df[c].astype(str).astype("category").cat.codes.values for c in CAT], 1).astype("int64")
        cards = [int(cats[:, i].max()) + 1 for i in range(len(CAT))]
        g_flat = torch.tensor(flat, device=DEV)
        g_num = torch.tensor(Xs, device=DEV)
        g_cat = torch.tensor(cats, device=DEV)
        g_pos = torch.tensor(df.pos.values, device=DEV)
        g_lb = torch.tensor(np.log(df.base.values).astype("float32"), device=DEV)
        g_y = torch.tensor(df.y.values.astype("float32"), device=DEV)
        offs = torch.arange(L - 1, -1, -1, device=DEV)

        def batch(idx):
            v = g_flat[g_pos[idx, None] - offs] - g_lb[idx, None]
            mask = ~torch.isnan(v)
            v = torch.nan_to_num(v, 0.0).clamp(-2, 2)
            return torch.stack([v, mask.float()], 2), g_num[idx], g_cat[idx]

        def predict(model, m):
            model.eval()
            ids = torch.tensor(np.where(m)[0], device=DEV)
            out = []
            with torch.no_grad():
                for i in range(0, len(ids), 16384):
                    out.append(model(*batch(ids[i:i + 16384])))
            return torch.cat(out).cpu().numpy()

        model = Net(Xs.shape[1], cards).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        tr_ids = torch.tensor(np.where(tr_m)[0], device=DEV)
        best, best_state, bad = 1e9, None, 0
        for ep in range(12):
            model.train()
            perm = tr_ids[torch.randperm(len(tr_ids), device=DEV)]
            for i in range(0, len(perm), 4096):
                idx = perm[i:i + 4096]
                loss = nn.functional.mse_loss(model(*batch(idx)), g_y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            va = np.sqrt(np.mean((predict(model, va_m) - df.y.values[va_m]) ** 2))
            print(f"  {nm} epoch {ep + 1}: val RMSE {va:.4f}", flush=True)
            if va < best - 1e-4:
                best, bad = va, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        model.load_state_dict(best_state)
        pred = predict(model, te_m)
        y = df.y.values[te_m].astype("float64")
        nat = df.is_nat.values[te_m] == 1
        for part, m in [("지역 테스트 행", ~nat), ("전국 테스트 행", nat)]:
            rows.append({"분야": nm, "평가 대상": part, "모델": "LSTM (시퀀스 + 피처)",
                         **{k: round(v, 4) for k, v in rm.metrics(y[m], pred[m].astype("float64")).items()}})
        print(nm, f"{time.time() - t:.0f}초", flush=True)
        print(pd.DataFrame([r for r in rows if r["분야"] == nm]).drop(columns="분야").to_string(index=False), flush=True)
        del df, g_flat, g_num, g_cat, g_pos, g_lb, g_y, model
        torch.cuda.empty_cache()
    r = pd.DataFrame(rows)
    comb = pd.read_csv(ROOT / "eda" / "tables" / "combined_model.csv")
    r = pd.concat([comb, r], ignore_index=True)
    r.to_csv(ROOT / "eda" / "tables" / "lstm_combined.csv", encoding="utf-8-sig", index=False)
    out = ["# LSTM vs 트리 통합 모델 (테스트 2023~2024)", "", "> 자동 생성: `python eda/15_lstm_combined.py`",
           "> 트리 통합 모델 결과는 eda/14_combined_model.py. LSTM은 2022-06~12를 조기 종료 검증셋으로 사용 (트리보다 학습 기간 6개월 짧음)", ""]
    for nm in r.분야.unique():
        out += [f"## {nm}", "", r[r.분야 == nm].drop(columns="분야").to_markdown(index=False, disable_numparse=True), ""]
    out.append(f"> 전체 소요 {time.time() - t_all:.0f}초")
    (ROOT / "eda" / "15_lstm_combined.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
