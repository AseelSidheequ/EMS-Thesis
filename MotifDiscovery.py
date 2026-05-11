import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import stumpy

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# SETTINGS
DATA_FILE        = "microgrid_results.csv"
MOTIF_WINDOW     = 24
RADIUS_MULT      = 2.0
MIN_RADIUS       = 0.30
CHAR_THRESHOLD   = 0.15

SIGNALS = [
    "battery_charge_kW", "battery_discharge_kW", "grid_buy_kW",
    "stored_energy_kWh", "price_buy_EUR_per_kWh", "pv_generation_kW",
]
NAMES = {
    "battery_charge_kW": "Battery Charge", "battery_discharge_kW": "Battery Discharge",
    "grid_buy_kW": "Grid Buy", "stored_energy_kWh": "Stored Energy",
    "price_buy_EUR_per_kWh": "Price Buy", "pv_generation_kW": "PV Generation",
}
MONTH_STARTS = [0,744,1416,2160,2880,3624,4344,5088,5832,6552,7296,8016,8760]
MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DAYS_PM      = [31,28,31,30,31,30,31,31,30,31,30,31]

# HELPERS
def month_of(h):
    for m in range(12):
        if MONTH_STARTS[m] <= h < MONTH_STARTS[m+1]:
            return m
    return 11

def znorm(s):
    return (s - s.mean()) / (s.std() + 1e-12)

# LOAD & COMPUTE MATRIX PROFILES
df = pd.read_csv(DATA_FILE)
SIGNALS = [s for s in SIGNALS if s in df.columns]
print(f"Loaded {DATA_FILE} | {len(df)} rows | {len(SIGNALS)} signals")

mp_all = {}
for col in SIGNALS:
    mp_all[col] = stumpy.stump(df[col].to_numpy(float), m=MOTIF_WINDOW)
    print(f"  MP done: {col}")

# CORE FUNCTIONS
def get_motif_discord(mp):
    """Return (motif_A, motif_B, motif_dist, discord_idx, discord_dist).
    Exclusion zone = m/2 per Yeh et al. 2016."""
    prof   = mp[:, 0].astype(float)
    nn     = mp[:, 1].astype(int)
    ez     = max(MOTIF_WINDOW // 2, 1)
    tmp    = prof.copy()
    ia     = int(np.argmin(tmp))
    ib     = int(nn[ia])
    mdist  = float(tmp[ia])
    for c in (ia, ib):
        tmp[max(0,c-ez):min(len(tmp),c+ez+1)] = np.inf
    disc   = int(np.argmax(np.where(np.isfinite(prof), prof, -np.inf)))
    return ia, ib, mdist, disc, float(prof[disc])

def motif_days(prof, mdist):
    """All non-overlapping days within the motif neighbourhood."""
    thr = max(RADIUS_MULT * mdist, MIN_RADIUS)
    ez  = max(MOTIF_WINDOW // 2, 1)
    cands, acc = np.where(prof <= thr)[0].tolist(), []
    for c in cands:
        if all(abs(c - a) >= ez for a in acc):
            acc.append(c)
    return sorted(set(h // MOTIF_WINDOW for h in acc))

# PLOT 1: MP OVERVIEW
fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(16, 3.5*len(SIGNALS)))
axes = np.array(axes).flatten()
t_ticks = [int((MONTH_STARTS[m]+MONTH_STARTS[m+1])/2) for m in range(12)]

for ax, col in zip(axes, SIGNALS):
    x    = df[col].to_numpy(float)
    mp   = mp_all[col]
    prof = mp[:,0].astype(float)
    ia, ib, mdist, disc, ddist = get_motif_discord(mp)

    ax2 = ax.twinx()
    ax.plot(x, color="#1565C0", lw=0.65, alpha=0.75)
    ax2.plot(prof, color="#E65100", lw=0.75, alpha=0.55)

    # Shaded spans (alpha raised so 24 h windows are visible at year scale)
    ax.axvspan(ia,   ia+MOTIF_WINDOW,   alpha=0.55, color="#43A047", zorder=0)
    ax.axvspan(ib,   ib+MOTIF_WINDOW,   alpha=0.55, color="#A5D6A7", zorder=0)
    ax.axvspan(disc, disc+MOTIF_WINDOW, alpha=0.55, color="#EF9A9A", zorder=0)

    # Vertical dashed lines at the centre of each window for reliable visibility
    ax.axvline(ia   + MOTIF_WINDOW//2, color="#43A047", lw=1.3, ls="--", alpha=0.95, zorder=2)
    ax.axvline(ib   + MOTIF_WINDOW//2, color="#2E7D32", lw=1.3, ls="--", alpha=0.95, zorder=2)
    ax.axvline(disc + MOTIF_WINDOW//2, color="#C62828", lw=1.3, ls="--", alpha=0.95, zorder=2)

    for ms in MONTH_STARTS[1:-1]:
        ax.axvline(ms, color="#BDBDBD", lw=0.5, ls=":", alpha=0.7)

    ax.set_xticks(t_ticks); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.set_ylabel(NAMES[col], fontsize=7.5, color="#1565C0")
    ax2.set_ylabel("MP dist", fontsize=7.5, color="#E65100")
    ax.tick_params(axis="y", labelcolor="#1565C0", labelsize=7)
    ax2.tick_params(axis="y", labelcolor="#E65100", labelsize=7)
    ax.grid(True, alpha=0.12)
    ma, mb, md = MONTH_NAMES[month_of(ia)], MONTH_NAMES[month_of(ib)], MONTH_NAMES[month_of(disc)]
    ax.set_title(
        f"{NAMES[col]} | Motif A=h{ia}({ma}) B=h{ib}({mb}) dist={mdist:.3f} | Discord h{disc}({md}) dist={ddist:.3f}",
        fontsize=8.5, fontweight="bold", pad=3)

# Legend: explains both lines and both window types
axes[0].legend(handles=[
    Line2D([0],[0], color="#1565C0", lw=1.5, label="Signal (left axis)"),
    Line2D([0],[0], color="#E65100", lw=1.5, label="MP distance (right axis)"),
    Patch(fc="#43A047", alpha=0.7, label="Motif A (24 h window)"),
    Patch(fc="#A5D6A7", alpha=0.7, label="Motif B (24 h window)"),
    Patch(fc="#EF9A9A", alpha=0.7, label="Discord (24 h window)"),
], fontsize=8, loc="upper right")

fig.suptitle(
    "Matrix Profile Overview — Motif and Discord Detection (Full Year)\n"
    "Window = 24 h | Exclusion zone = m/2",
    fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.97])
plt.savefig("output/fig_motif_mp_overview.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_motif_mp_overview.png")

# PLOT 2: MOTIF GALLERY
n_cols = 3
fig, axes = plt.subplots(int(np.ceil(len(SIGNALS)/n_cols)), n_cols,
                          figsize=(15, 5*int(np.ceil(len(SIGNALS)/n_cols))),
                          sharey=True)
axes = np.array(axes).flatten()
t24  = np.arange(MOTIF_WINDOW)

for ax, col in zip(axes, SIGNALS):
    x = df[col].to_numpy(float)
    ia, ib, mdist, disc, ddist = get_motif_discord(mp_all[col])
    ma, mb, md = MONTH_NAMES[month_of(ia)], MONTH_NAMES[month_of(ib)], MONTH_NAMES[month_of(disc)]

    ax.plot(t24, znorm(x[ia:ia+MOTIF_WINDOW]),    color="#1E88E5", lw=2.0,        label=f"Motif A (h{ia},{ma})")
    ax.plot(t24, znorm(x[ib:ib+MOTIF_WINDOW]),    color="#43A047", lw=2.0, ls="--", label=f"Motif B (h{ib},{mb})")
    ax.plot(t24, znorm(x[disc:disc+MOTIF_WINDOW]), color="#E53935", lw=1.4, ls=":", alpha=0.85, label=f"Discord (h{disc},{md})")
    ax.axhline(0, color="#BDBDBD", lw=0.5, alpha=0.5)
    ax.set_xlim(0, MOTIF_WINDOW-1); ax.set_xticks(range(0,MOTIF_WINDOW,4))
    ax.set_title(f"{NAMES[col]}\nMotif dist = {mdist:.3f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Hour", fontsize=8); ax.set_ylabel("z-score", fontsize=8)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.18)

for j in range(len(SIGNALS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Top-1 Motif Pair vs Discord — z-normalised 24 h Shape\n",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.95])
plt.savefig("output/fig_motif_gallery.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_motif_gallery.png")

# PLOT 3: MOTIF CALENDAR
CAL_SIGNALS = [s for s in ["battery_charge_kW","grid_buy_kW"] if s in SIGNALS]
fig, axes = plt.subplots(len(CAL_SIGNALS), 1, figsize=(15, 4.8*len(CAL_SIGNALS)))
axes = np.array(axes).flatten()

for ax, col in zip(axes, CAL_SIGNALS):
    mp   = mp_all[col]
    prof = mp[:,0].astype(float)
    ia, ib, mdist, disc, _ = get_motif_discord(mp)
    mdays    = motif_days(prof, mdist)
    disc_day = disc // MOTIF_WINDOW

    label = np.zeros(365)
    for d in mdays:
        if d < 365: label[d] = 1
    if disc_day < 365: label[disc_day] = -1

    grid, day = np.full((12,31), np.nan), 0
    for m, nd in enumerate(DAYS_PM):
        for d in range(nd):
            if day < 365: grid[m,d] = label[day]
            day += 1

    ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", interpolation="none")
    ax.set_yticks(range(12)); ax.set_yticklabels(MONTH_NAMES, fontsize=9)
    ax.set_xticks(range(31)); ax.set_xticklabels([str(d+1) for d in range(31)], fontsize=6.5)
    ax.set_xlabel("Day of month", fontsize=8.5)

    total, day = 0, 0
    for m, nd in enumerate(DAYS_PM):
        cnt = sum(1 for d in range(nd) if day+d < 365 and label[day+d] == 1)
        total += cnt
        ax.text(31.6, m, f"{cnt}/{nd}\n({cnt/nd*100:.0f}%)", va="center", fontsize=6.5, color="#1B5E20")
        day += nd

    flag = "YES" if total/365 >= CHAR_THRESHOLD else "NO"
    ax.text(31.6, 12, f"TOTAL\n{total}/365\n({total/365*100:.0f}%)\nCharacteristic: {flag}",
            va="top", fontsize=7, color="#1B5E20", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#1B5E20", alpha=0.9))

    # Colorbar removed — colour meaning encoded directly in title to avoid
    # misleading readers into treating the categorical -1/0/+1 labels as continuous values
    ax.set_title(
        f"{NAMES[col]} — Motif Occurrence Calendar\n"
        "Green = strategy-following day  |  Red = discord (most anomalous)  |  Yellow = normal",
        fontsize=9.5, fontweight="bold")

fig.suptitle("Motif Occurrence Calendar\nCharacteristic threshold = 15%",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.95])
plt.savefig("output/fig_motif_calendar.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_motif_calendar.png")

# PLOT 4: DISCORD CONTEXT
CTX = [s for s in ["battery_charge_kW","battery_discharge_kW","grid_buy_kW"] if s in SIGNALS]
fig, axes = plt.subplots(1, len(CTX), figsize=(6.5*len(CTX), 5.2), sharey=True)
axes = np.array(axes).flatten()

for ax, col in zip(axes, CTX):
    x  = df[col].to_numpy(float)
    ia, ib, mdist, disc, ddist = get_motif_discord(mp_all[col])

    daily = np.array([znorm(x[d*MOTIF_WINDOW:(d+1)*MOTIF_WINDOW]) for d in range(len(x)//MOTIF_WINDOW)])
    med   = np.median(daily, axis=0)
    std   = daily.std(axis=0)
    t24   = np.arange(MOTIF_WINDOW)

    ax.fill_between(t24, med-std, med+std, alpha=0.12, color="#9E9E9E", label="±1σ all days")
    ax.plot(t24, med,                           color="#9E9E9E", lw=1.5, ls="--", label="Median shape")
    ax.plot(t24, znorm(x[ia:ia+MOTIF_WINDOW]),  color="#1E88E5", lw=2.0, label=f"Top motif (h{ia})")
    ax.plot(t24, znorm(x[disc:disc+MOTIF_WINDOW]), color="#E53935", lw=2.2,
            label=f"Discord (h{disc},{MONTH_NAMES[month_of(disc)]})")

    h_twin, l_twin = [], []
    if "price_buy_EUR_per_kWh" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(t24, df["price_buy_EUR_per_kWh"].iloc[disc:disc+MOTIF_WINDOW].values*100,
                 color="#7B1FA2", lw=1.2, ls=":", alpha=0.85, label="Price discord day (c€/kWh)")
        ax2.set_ylabel("Price (c€/kWh)", fontsize=8, color="#7B1FA2")
        ax2.tick_params(axis="y", labelcolor="#7B1FA2", labelsize=7.5)
        h_twin, l_twin = ax2.get_legend_handles_labels()

    ax.set_xlim(0, MOTIF_WINDOW-1); ax.set_xticks(range(0,MOTIF_WINDOW,4))
    ax.set_xlabel("Hour", fontsize=8.5); ax.set_ylabel("z-score", fontsize=8.5)
    ax.set_title(f"{NAMES[col]}\nDiscord dist = {ddist:.3f}", fontsize=9.5, fontweight="bold")
    ax.grid(True, alpha=0.18)
    h_main, l_main = ax.get_legend_handles_labels()
    ax.legend(h_main+h_twin, l_main+l_twin, fontsize=7.5, loc="upper left")

fig.suptitle("Discord vs Median Daily Shape — Anomalous Dispatch Days\n",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0,0.01,1,0.91])
plt.savefig("output/fig_motif_discord_context.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: fig_motif_discord_context.png")

# SUMMARY
print("\n Summary")
for col in SIGNALS:
    ia, ib, mdist, disc, ddist = get_motif_discord(mp_all[col])
    print(f"{col:<28} Motif A=h{ia}({MONTH_NAMES[month_of(ia)]}) "
          f"B=h{ib}({MONTH_NAMES[month_of(ib)]}) dist={mdist:.3f} | "
          f"Discord h{disc}({MONTH_NAMES[month_of(disc)]}) dist={ddist:.3f}")
print("\nDone.")