# 예측 확률 식에 '높게 말하면 더 큰 벌점'(틀린 행 가중치 k배)을 주고 검증으로 맞춘 뒤 테스트에서 확인 → eda/figures/direction_prob_asym.png
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = open(__file__.replace("76_direction_prob_asym.py", "73_direction_prob.py"), encoding="utf-8").read().split('if __name__ == "__main__":')[0]
exec(src.replace("Path(__file__).resolve().parent.parent", "Path(r'" + __file__.replace("\\", "/").rsplit("/eda/", 1)[0] + "')"))
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
K = [1, 1.2, 1.5, 2]
COL = {1: "#d1495b", 1.2: "#edae49", 1.5: "#66a182", 2: "#00798c"}


def wfit(x, h, k):
    """p = a + b·ln x, 오차 제곱합에서 틀린 행(h=0, 식이 높게 말한 쪽)에 k배 가중."""
    w = np.where(h == 0, k, 1.0)
    b, a = np.polyfit(np.log(x), h, 1, w=np.sqrt(w))
    return a, b


def ev(p, h):
    p = np.minimum(np.clip(p, 0, 1), CAP); g = []
    for lo in np.arange(0.5, 0.85, 0.05):
        m = (p > lo) & (p <= lo + 0.05)
        if m.sum() > 500:
            g.append(p[m].mean() - h[m].mean())
    g = np.array(g)
    return np.mean((p - h) ** 2), g[np.abs(g).argmax()], (p <= 0.5).mean()


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (d, nm) in zip(axes, NAMES.items()):
    yv, pv, yt, pt = load(d); sv, hv = hits(pv, yv); st, ht = hits(pt, yt)
    edges = np.quantile(st, np.linspace(0, 1, 51)); b_ = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, 49)
    mid = [np.median(st[b_ == i]) * 100 for i in range(50)]; act = [ht[b_ == i].mean() * 100 for i in range(50)]
    ax.plot(mid, act, "o", color="#555", ms=3.5, label="테스트 실제 적중률 (50구간)")
    g = np.linspace(0.01, 0.30, 300)
    print(f"== {nm}")
    for k in K:
        a, b = wfit(sv, hv, k)
        br, gap, warn = ev(a + b * np.log(st), ht)
        print(f"  벌점 {k}배: a={a:.3f} b={b:.3f}  테스트 Brier {br:.4f}  가장 큰 차이(표시−실제) {gap:+.1%}p  주의 문구 비율 {warn:.0%}")
        ax.plot(g * 100, np.minimum(np.clip(a + b * np.log(g), 0, 1), CAP) * 100, color=COL[k], lw=2 if k == 1 else 1.5,
                label=f"높게 말한 오차 ×{k}" + (" (지금)" if k == 1 else ""))
    ax.axhline(50, color="#999", lw=0.8, ls=":")
    ax.set_xlim(0, 30); ax.set_ylim(25, 95); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(nm); ax.set_xlabel("예측 변화 크기 (절댓값)")
axes[0].set_ylabel("예측 방향으로 1% 넘게 움직일 확률 (%)")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("검증(2023~24)으로 맞춘 곡선 vs 테스트(2025~26) 실제", fontsize=11)
fig.tight_layout(); fig.savefig(F / "direction_prob_asym.png", dpi=150)
print("저장: eda/figures/direction_prob_asym.png")
