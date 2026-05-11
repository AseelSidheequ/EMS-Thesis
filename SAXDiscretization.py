import os, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# SETTINGS
DATA_FILE  = "microgrid_results.csv"
SAX_ALPHA  = 5
SAX_WINDOW = 24
SAX_W      = 8

SIGNALS = [
    "stored_energy_kWh","price_buy_EUR_per_kWh","pv_generation_kW","load_demand_kW",
    "battery_charge_kW","battery_discharge_kW","grid_buy_kW","grid_sell_kW",
]
NAMES = {
    "stored_energy_kWh":"SoC","price_buy_EUR_per_kWh":"Price Buy","pv_generation_kW":"PV Generation",
    "load_demand_kW":"Load Demand","battery_charge_kW":"Battery Charge",
    "battery_discharge_kW":"Battery Discharge","grid_buy_kW":"Grid Buy","grid_sell_kW":"Grid Sell",
}
LETTERS       = [chr(ord("a")+k) for k in range(SAX_ALPHA)]
LETTER_LABELS = ["very low","low","medium","high","very high"]
LETTER_COLORS = ["#1565C0","#2E7D32","#F9A825","#C62828","#6A1B9A"]
BLOCK_HOURS   = [f"{3*i:02d}-{3*(i+1):02d}h" for i in range(SAX_W)]
MONTH_STARTS  = [0,744,1416,2160,2880,3624,4344,5088,5832,6552,7296,8016,8760]
MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DAYS_PM       = [31,28,31,30,31,30,31,31,30,31,30,31]

# SAX CORE
BREAKS = norm.ppf(np.linspace(1/SAX_ALPHA, 1-1/SAX_ALPHA, SAX_ALPHA-1))

def znorm(x):
    return (x - x.mean()) / (x.std() + 1e-12)

def paa(x, w):
    edges = np.linspace(0, len(x), w+1)
    return np.array([x[int(np.floor(edges[k])):max(int(np.ceil(edges[k+1])),int(np.floor(edges[k]))+1)].mean()
                     for k in range(w)])

def sax_encode(x):
    return "".join(LETTERS[np.searchsorted(BREAKS, v)] for v in paa(znorm(x), SAX_W))

def encode_year(series):
    n_days = len(series) // SAX_WINDOW
    return [(d, d*SAX_WINDOW, sax_encode(series[d*SAX_WINDOW:(d+1)*SAX_WINDOW]))
            for d in range(n_days)]

# LOAD & ENCODE
df = pd.read_csv(DATA_FILE)
SIGNALS = [s for s in SIGNALS if s in df.columns]
print(f"Loaded {DATA_FILE} | {len(df)} rows")

sax = {col: encode_year(df[col].to_numpy(float)) for col in SIGNALS}

day_to_month = []
for m, nd in enumerate(DAYS_PM):
    day_to_month.extend([m]*nd)

# PLOT 1: SYMBOL FREQUENCY
# sharey=True: shared axis set by matplotlib auto-scaling to the global max across all subplots.
# ax.set_ylim removed from the loop — individual calls with sharey active override each other
# (last subplot wins by accident); auto-scaling is the correct behaviour here.
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
for ax, col in zip(axes.flatten(), SIGNALS):
    chars = "".join(w for _,_,w in sax[col])
    freq  = Counter(chars)
    pcts  = [freq.get(l,0)/len(chars)*100 for l in LETTERS]
    bars  = ax.bar(LETTERS, pcts, color=LETTER_COLORS, alpha=0.82, edgecolor="white")
    ax.axhline(100/SAX_ALPHA, color="#212121", ls="--", lw=1.1, label=f"Equi-prob ({100/SAX_ALPHA:.0f}%)")
    for bar, p in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, p+0.3, f"{p:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title(NAMES[col], fontsize=9, fontweight="bold", pad=3)
    ax.set_xlabel("SAX symbol", fontsize=8); ax.set_ylabel("Frequency (%)", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.18)

fig.suptitle(f"SAX Symbol Frequency (A={SAX_ALPHA}, W={SAX_W}, {SAX_WINDOW//SAX_W} h/letter)\n"
             "Dashed = equi-probable baseline", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.92])
plt.savefig("output/fig_sax_frequency.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_sax_frequency.png")

# PLOT 2: SAX OVERLAY ON PRICE
if "price_buy_EUR_per_kWh" in df.columns:
    col   = "price_buy_EUR_per_kWh"
    x_n   = znorm(df[col].to_numpy(float))
    t_mid = [int((MONTH_STARTS[m]+MONTH_STARTS[m+1])/2) for m in range(12)]
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(x_n, color="#B0BEC5", lw=0.55, alpha=0.65, label="z-normalised")

    for day, start, word in sax[col]:
        fs = SAX_WINDOW / SAX_W
        for k, letter in enumerate(word):
            t0 = start + int(k*fs); t1 = max(start + int((k+1)*fs), t0+1)
            ax.hlines(x_n[t0:t1].mean(), t0, t1-1,
                      colors=LETTER_COLORS[LETTERS.index(letter)], lw=3.0, alpha=0.9)
        if day % 30 == 0:
            mid = start + SAX_WINDOW//2
            ax.text(mid, x_n[start:start+SAX_WINDOW].max()+0.2, word, ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#90A4AE", alpha=0.85))

    for ms in MONTH_STARTS[1:-1]:
        ax.axvline(ms, color="#B0BEC5", lw=0.6, ls=":", alpha=0.8)
    ax.set_xticks(t_mid); ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_title(f"Price Buy — SAX Overlay (A={SAX_ALPHA}, W={SAX_W}, {SAX_WINDOW//SAX_W} h/letter)",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Month", fontsize=10); ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.12)
    ax.legend(handles=[Line2D([0],[0],color=LETTER_COLORS[k],lw=3,label=f"'{LETTERS[k]}' {LETTER_LABELS[k]}")
                        for k in range(SAX_ALPHA)], fontsize=8, loc="upper right", ncol=SAX_ALPHA)
    plt.tight_layout()
    plt.savefig("output/fig_sax_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Saved: fig_sax_overlay.png")

# PLOT 3: TOP-5 WORD SHAPES
KEY_TW = [s for s in ["battery_charge_kW","battery_discharge_kW","grid_buy_kW","stored_energy_kWh"] if s in SIGNALS]
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
t = np.arange(SAX_WINDOW)

for ax, col in zip(axes.flatten(), KEY_TW):
    x     = df[col].to_numpy(float)
    words = [w for _,_,w in sax[col]]
    top5  = Counter(words).most_common(5)
    cols5 = plt.cm.viridis(np.linspace(0.1, 0.88, len(top5)))

    for (word, cnt), c in zip(top5, cols5):
        segs = [znorm(x[s:s+SAX_WINDOW]) for _,s,w in sax[col]
                if w == word and len(x[s:s+SAX_WINDOW]) == SAX_WINDOW]
        if not segs: continue
        arr = np.array(segs)
        ax.plot(t, arr.mean(axis=0), lw=2.0, color=c,
                label=f'#{list(dict(top5).keys()).index(word)+1}: "{word}" ({cnt}d, {cnt/len(words)*100:.1f}%)')
        ax.fill_between(t, arr.mean(axis=0)-arr.std(axis=0), arr.mean(axis=0)+arr.std(axis=0),
                        alpha=0.08, color=c)

    ax.axhline(0, color="#BDBDBD", lw=0.6, alpha=0.5)
    ax.set_xlim(0, SAX_WINDOW-1); ax.set_xticks(range(0,SAX_WINDOW,4))
    ax.set_title(f"{NAMES[col]}\nTop-5 {SAX_W}-letter words", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Hour", fontsize=8.5); ax.set_ylabel("z-score", fontsize=8.5)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.18)

fig.suptitle("Top-5 Most Frequent Daily SAX Words | ±1σ bands | 3 h/letter\n"
             "Characteristic motif threshold = 15%",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.93])
plt.savefig("output/fig_sax_topwords.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_sax_topwords.png")

# PLOT 4: MONTHLY WORD EVOLUTION
KEY_MO = [s for s in ["battery_charge_kW","grid_buy_kW"] if s in SIGNALS]
fig, axes = plt.subplots(1, len(KEY_MO), figsize=(17, 5.5), sharey=True)
axes = np.array(axes).flatten()

for ax, col in zip(axes, KEY_MO):
    all_words = [w for _,_,w in sax[col]]
    top5      = [w for w,_ in Counter(all_words).most_common(5)]
    mcounts   = {w: np.zeros(12) for w in top5}
    mtotals   = np.zeros(12)
    for day,_,word in sax[col]:
        if day >= len(day_to_month): continue
        mi = day_to_month[day]
        if word in mcounts: mcounts[word][mi] += 1
        mtotals[mi] += 1

    bottom = np.zeros(12)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top5)))
    for w, c in zip(top5, colors):
        pcts = np.divide(mcounts[w], mtotals, out=np.zeros(12), where=mtotals>0)*100
        ax.bar(range(12), pcts, bottom=bottom, color=c, alpha=0.85,
               label=f'"{w}"', edgecolor="white", lw=0.5)
        for m in range(12):
            if pcts[m] > 8:
                ax.text(m, bottom[m]+pcts[m]/2, f"{int(mcounts[w][m])}d",
                        ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
        bottom += pcts

    ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_ylabel("Share of days (%)", fontsize=9); ax.set_ylim(0, 115)
    ax.set_title(f"{NAMES[col]}\nMonthly top-5 word frequency", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", title="SAX word", title_fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.18)

fig.suptitle("Seasonal SAX Word Evolution", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.92])
plt.savefig("output/fig_sax_monthly_evolution.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_sax_monthly_evolution.png")

# PLOT 5: TEMPORAL ASSOCIATION RULES
# Confidence(X='e' at block i → Y='e' at block j) — Funde et al. 2019, Eq.(3)
KEY_PAIRS = [(a, b, d) for a, b, d in [
    ("price_buy_EUR_per_kWh","battery_discharge_kW","Price→Discharge"),
    ("battery_charge_kW",    "battery_discharge_kW","Charge→Discharge"),
] if a in sax and b in sax]

if KEY_PAIRS:
    fig, axes = plt.subplots(1, len(KEY_PAIRS), figsize=(7*len(KEY_PAIRS), 7))
    axes = np.array(axes).flatten()

    for ax, (col_a, col_b, desc) in zip(axes, KEY_PAIRS):
        seq_a = [[word[i] for i in range(SAX_W)] for _,_,word in sax[col_a]]
        seq_b = [[word[i] for i in range(SAX_W)] for _,_,word in sax[col_b]]
        nd    = len(seq_a)
        conf  = np.zeros((SAX_W, SAX_W))

        for i in range(SAX_W):
            for j in range(SAX_W):
                ant   = sum(1 for d in range(nd) if seq_a[d][i] == "e")
                joint = sum(1 for d in range(nd) if seq_a[d][i] == "e" and seq_b[d][j] == "e")
                conf[i,j] = joint/ant if ant > 0 else 0.0

        im = ax.imshow(conf, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto", origin="upper", interpolation="none")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f"P({NAMES[col_b]}='e' at j | {NAMES[col_a]}='e' at i)")
        for i in range(SAX_W):
            for j in range(SAX_W):
                v = conf[i,j]
                if v > 0.08:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v > 0.60 else "#333333",
                            fontweight="bold" if v >= 0.70 else "normal")
                if j > i and v >= 0.70:
                    ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,ec="#1565C0",lw=2.0,zorder=3))

        ax.plot(range(SAX_W), range(SAX_W), "w--", lw=0.9, alpha=0.5)
        ax.set_xticks(range(SAX_W)); ax.set_xticklabels(BLOCK_HOURS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(SAX_W)); ax.set_yticklabels(BLOCK_HOURS, fontsize=8)
        ax.set_xlabel(f"{NAMES[col_b]} block j", fontsize=9)
        ax.set_ylabel(f"{NAMES[col_a]} block i", fontsize=9)
        ax.set_title(f"{desc}\nBlue border: forward-lag rules (j>i, conf≥0.70)",
                     fontsize=9.5, fontweight="bold")

    fig.suptitle("Temporal Association Rules — Phase-Lag Relationships\n"
                 "Symbol 'e' = very high",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0,0.01,1,0.90])
    plt.savefig("output/fig_sax_assoc_rules.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Saved: fig_sax_assoc_rules.png")

# PLOT 6: SAX–MP BRIDGE
try:
    import stumpy as _st
    BRIDGE_COL = "battery_discharge_kW"
    if BRIDGE_COL in df.columns:
        x_br = df[BRIDGE_COL].to_numpy(float)
        prof = _st.stump(x_br, m=SAX_WINDOW)[:,0].astype(float)
        ez   = max(SAX_WINDOW//2, 1)
        ia   = int(np.argmin(prof))
        bd   = float(prof[ia])
        rad  = max(2.0*bd, 0.30)

        cands, acc = np.where(prof <= rad)[0].tolist(), []
        for c in cands:
            if all(abs(c-a) >= ez for a in acc): acc.append(c)
        mdays = set(h//SAX_WINDOW for h in acc)

        wm  = [w for d,_,w in sax[BRIDGE_COL] if d in mdays]
        wnm = [w for d,_,w in sax[BRIDGE_COL] if d not in mdays]
        nm, nnm = len(wm), len(wnm)
        top8 = [w for w,_ in Counter(wm+wnm).most_common(8)]
        fm, fnm = Counter(wm), Counter(wnm)
        pm  = [fm.get(w,0)/max(nm,1)*100  for w in top8]
        pnm = [fnm.get(w,0)/max(nnm,1)*100 for w in top8]

        fig, ax = plt.subplots(figsize=(12, 5.5))
        xp, wd = np.arange(len(top8)), 0.38
        bm  = ax.bar(xp-wd/2, pm,  wd, color="#1B5E20", alpha=0.82, label=f"Motif days (n={nm})")
        bnm = ax.bar(xp+wd/2, pnm, wd, color="#9E9E9E", alpha=0.65, label=f"Non-motif (n={nnm})")
        for bar, p in zip(bm, pm):
            if p > 1: ax.text(bar.get_x()+bar.get_width()/2, p+0.3, f"{p:.1f}%",
                              ha="center", va="bottom", fontsize=8, color="#1B5E20", fontweight="bold")
        for bar, p in zip(bnm, pnm):
            if p > 1: ax.text(bar.get_x()+bar.get_width()/2, p+0.3, f"{p:.1f}%",
                              ha="center", va="bottom", fontsize=8, color="#616161")

        ax.set_xticks(xp); ax.set_xticklabels([f'"{w}"' for w in top8], fontsize=8.5, rotation=15, ha="right")
        ax.set_ylabel("Share of days in group (%)", fontsize=9.5)
        ax.set_title(f"{NAMES[BRIDGE_COL]} — SAX Word Enrichment on Motif Days\n"
                     f"MP radius = {rad:.2f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.18)
        plt.tight_layout()
        plt.savefig("output/fig_sax_motif_bridge.png", dpi=150, bbox_inches="tight")
        plt.close(); print("Saved: fig_sax_motif_bridge.png")
except ImportError:
    print("stumpy not available — skipping SAX–MP bridge.")

# TOP-5 WORD CONSOLE SUMMARY
print("\n Top-5 SAX Words per Signal ")
for col in SIGNALS:
    words = [w for _,_,w in sax[col]]
    print(f"\n {NAMES[col]} ({len(words)}d):")
    for rank,(word,cnt) in enumerate(Counter(words).most_common(5),1):
        print(f'  #{rank}: "{word}" {cnt}d ({cnt/len(words)*100:.1f}%)')
print("\nDone.")