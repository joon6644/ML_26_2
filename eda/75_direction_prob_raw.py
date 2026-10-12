# 테스트(2025~26) 실제 적중률(50구간) vs 카드 확률 식 a + b·ln x — 원본 눈금(로그 곡선) → eda/figures/direction_prob_raw.png
import numpy as np, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
src = open(__file__.replace("75_direction_prob_raw.py", "73_direction_prob.py"), encoding="utf-8").read().split('if __name__ == "__main__":')[0]
exec(src.replace("Path(__file__).resolve().parent.parent", "Path(r'" + __file__.replace("\\", "/").rsplit("/eda/", 1)[0] + "')"))
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
rng = np.random.default_rng(0)
NB = 50
for ax, (d, nm) in zip(axes, NAMES.items()):
    yv, pv, yt, pt = load(d); st, ht = hits(pt, yt)
    c = joblib.load(M / f"prob_{d}.joblib")
    x = st * 100
    edges = np.quantile(st, np.linspace(0, 1, NB + 1)); b = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, NB - 1)
    mid = np.array([np.median(st[b == i]) for i in range(NB)]) * 100
    act = np.array([ht[b == i].mean() for i in range(NB)])
    se = np.sqrt(act * (1 - act) / (len(st) / NB))
    ax.errorbar(mid, act * 100, yerr=196 * se, fmt="o", ms=3.5, color="#2e86ab", elinewidth=0.8)
    g = np.geomspace(0.01, st.max(), 300); raw = np.clip(c["a"] + c["b"] * np.log(g), 0, 1)
    ax.plot(g * 100, np.minimum(raw, 0.85) * 100, color="#d1495b", lw=2)
    ax.plot(g * 100, raw * 100, color="#d1495b", lw=1, ls="--")
    ax.axhline(50, color="#999", lw=0.8, ls=":")
    ax.set_xlim(0, 30); ax.set_ylim(20, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(f"{nm} — 테스트 {len(st):,}행 (구간당 약 {len(st) // NB:,}행)", fontsize=10)
    ax.set_xlabel("예측 변화 크기 (절댓값)")
axes[0].set_ylabel("예측 방향으로 1% 넘게 움직인 비율 (%)")
axes[0].legend(handles=[
    Line2D([], [], ls="", marker="o", color="#2e86ab", ms=4, label=f"실제 적중률 ({NB}구간, 95% 신뢰구간)"),
    Line2D([], [], color="#d1495b", lw=2, label="카드 확률 a + b·ln x (최대 85%)"),
    Line2D([], [], color="#d1495b", lw=1, ls="--", label="상한이 없을 때")], frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.1))
fig.tight_layout(); fig.savefig(F / "direction_prob_raw.png", dpi=150)
print("저장: eda/figures/direction_prob_raw.png")
