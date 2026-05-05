import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
import os
import warnings

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# SETTINGS
DATA_FILE = 'microgrid_results.csv'
WAVELET = 'db4'
DWT_LEV = 10

SIGNALS = [
    'price_buy_EUR_per_kWh',
    'load_demand_kW',
    'pv_generation_kW',
    'grid_buy_kW',
    'grid_sell_kW',
    'battery_charge_kW',
    'battery_discharge_kW',
    'stored_energy_kWh'
]

SIMPLE_NAMES = {
    'price_buy_EUR_per_kWh': 'Price buy',
    'load_demand_kW': 'Load demand',
    'pv_generation_kW': 'PV generation',
    'grid_buy_kW': 'Grid buy',
    'grid_sell_kW': 'Grid sell',
    'battery_charge_kW': 'Battery charge',
    'battery_discharge_kW': 'Battery discharge',
    'stored_energy_kWh': 'Stored energy'
}

FFT_MARKERS = {
    8: '8h',
    12: '12h',
    24: '24h',
    48: '48h',
    168: '168h',
    720: '720h',
    8760: '8760h'
}

# LOAD DATA
df = pd.read_csv(DATA_FILE)
SIGNALS = [s for s in SIGNALS if s in df.columns]
N = len(df)

print(f'Loaded: {DATA_FILE} ({N} rows x {len(df.columns)} columns)')
print(f'Analysing {len(SIGNALS)} signals: {SIGNALS}')

# FFT

window = np.hanning(N) # [Taper edges to mitigate false discontinuities/ripples at block boundaries]
win_factor = np.sum(window ** 2) / N # [Scaling factor to balance ripple attenuation vs. spectral resolution trade-off]
fft_spectra = {}

for col in SIGNALS:
    x = df[col].to_numpy(dtype=float)
    x = x - x.mean() # [Remove DC-offset to focus spectrum on signal fluctuations]

    X = np.fft.rfft(x * window) # [Compute FFT for one-to-one transformation into frequency-domain]
    frq = np.fft.rfftfreq(N, d=1.0) # [Produce N/2 unique frequency bins from zero to nyquist frequency]
    pwr = (np.abs(X) ** 2) / (N * win_factor) # [Estimate power spectrum using squared FFT magnitude]
    ptot = pwr[1:].sum() + 1e-12 # [Sum total signal power to enable percentage-based spectral estimation]

    pers = 1.0 / frq[frq > 0] # [Map frequency bins to period domain]
    pw = pwr[frq > 0] / ptot * 100 # [Convert to percentage contribution for cross-signal comparison]
    valid = (pers >= 2) & (pers <= 8760) # [Filter to physically resolvable range (Nyquist limit to observation length)]

    fft_spectra[col] = (pers[valid], pw[valid])

# DWT
dwt_energies = {}

for col in SIGNALS:
    x = df[col].to_numpy(dtype=float)
    x = x - x.mean()

    # [Represent time series as a sum of time-limited oscillatory pulses (mother wavelet: db4)]
    coeffs = pywt.wavedec(x, WAVELET, level=DWT_LEV) 
    total = sum(np.sum(c ** 2) for c in coeffs) + 1e-12 # [Calculate total signal energy for decomposition]
    lev_data = []

    # Detail bands (time-frequency decomposition)
    for lev_idx in range(1, len(coeffs)):
        k = len(coeffs) - lev_idx
        period = 2 ** k
        epct = np.sum(coeffs[lev_idx] ** 2) / total * 100 #[Decompose into time scaled/shifted wavelet pulses]
        lev_data.append({
            'period': period,
            'energy': epct,
            'label': f'{period}h'
        })
    lev_data = sorted(lev_data, key=lambda d: d['period'], reverse=False)
    lev_data.append({
        'period': 9999,  # dummy value, never used for plotting
        'energy': np.sum(coeffs[0] ** 2) / total * 100,
        'label': 'Trend\n(>1024h)'
    })

    dwt_energies[col] = lev_data

# PLOT 1: FFT
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(4, 2, hspace=0.70, wspace=0.30)

for i, col in enumerate(SIGNALS):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    pv, pw = fft_spectra[col]

    ax.fill_between(pv, pw, color='#7B1FA2', alpha=0.10)
    ax.plot(pv, pw, color='#7B1FA2', linewidth=1.2)

    ax.set_xscale('log')
    ax.set_xlim(8, 8760)
    ax.set_xticks(list(FFT_MARKERS.keys()))
    ax.set_xticklabels(list(FFT_MARKERS.values()), fontsize=7, rotation=30, ha='right')

    ymax = max(pw) if len(pw) > 0 else 1.0
    for xp, lbl in FFT_MARKERS.items():
        ax.axvline(x=xp, color='gray', linestyle='--', linewidth=0.8, alpha=0.45)
        if xp < 8760:
            ax.text(
                xp, ymax * 0.88, lbl,
                fontsize=6.5, color='dimgray', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.70)
            )

    ax.set_xlabel('Period (hours, log scale)', fontsize=8)
    ax.set_ylabel('Power (%)', fontsize=8)
    ax.set_title(SIMPLE_NAMES[col], fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, which='major', alpha=0.20)
    ax.grid(True, which='minor', alpha=0.08)

fig.suptitle('FFT Power Spectrum', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('output/fig_spectral_fft.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: output/fig_spectral_fft.png')

# PLOT 2: DWT
fig2 = plt.figure(figsize=(14, 12))
gs2 = gridspec.GridSpec(4, 2, hspace=0.70, wspace=0.30)

for i, col in enumerate(SIGNALS):
    ax = fig2.add_subplot(gs2[i // 2, i % 2])
    lev_data = dwt_energies[col]

    xlabels = [d['label'] for d in lev_data]
    yvals   = [d['energy'] for d in lev_data]
    n       = len(xlabels)

    ax.bar(range(n), yvals, color='#1E88E5', alpha=0.75)
    ax.set_xticks(range(n))
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
    ax.set_xlim(-0.5, n - 0.5)

    ax.set_xlabel('Temporal scale', fontsize=8)
    ax.set_ylabel('Energy (%)', fontsize=8)
    ax.set_title(SIMPLE_NAMES[col], fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, axis='y', alpha=0.20)

fig2.suptitle('DWT Energy Distribution', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
plt.savefig('output/fig_spectral_dwt.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: output/fig_spectral_dwt.png')



# POINT 1 : Price Buy – sample daily plots
# POINT 2 : Grid / Battery – one day per month

PRICE_SAMPLE_DAYS = [
    ('Mar',  2),  
    ('Mar', 14), 
    ('Sep',  2),
    ('Sep', 14),   
]

SYNC_REP_DAY = 14   # Day 14 of every month

_MONTH_STARTS = {
    'Jan': 0,    'Feb': 744,  'Mar': 1416, 'Apr': 2160,
    'May': 2880, 'Jun': 3624, 'Jul': 4344, 'Aug': 5088,
    'Sep': 5832, 'Oct': 6552, 'Nov': 7296, 'Dec': 8016,
}

def _day_start(month_abbr, day_1indexed):
    """Return the absolute hour index for the start of a given day."""
    return _MONTH_STARTS[month_abbr] + (day_1indexed - 1) * 24

# PLOT 3 – Price Buy

n_days  = len(PRICE_SAMPLE_DAYS)
n_cols  = 2
n_rows  = (n_days + 1) // n_cols

_PRICE_COLOURS = ['#1565C0', '#42A5F5', '#BF360C', '#FF7043',
                  '#2E7D32', '#66BB6A', '#6A1B9A', '#AB47BC']

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharey=True)
axes3 = np.array(axes3).flatten()

hours_24 = np.arange(24)

for i, (month, day) in enumerate(PRICE_SAMPLE_DAYS):
    ax  = axes3[i]
    col = _PRICE_COLOURS[i % len(_PRICE_COLOURS)]
    s   = _day_start(month, day)
    price = df['price_buy_EUR_per_kWh'].iloc[s:s + 24].values

    ax.fill_between(hours_24, price, alpha=0.12, color=col)
    ax.plot(hours_24, price, color=col, linewidth=2.0,
            marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel('Hour of day', fontsize=8.5)
    ax.set_ylabel('Price (EUR/MWh)', fontsize=8.5)
    ax.set_title(f'{month} – Day {day}', fontsize=10, fontweight='bold', pad=6)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.22)

for j in range(n_days, len(axes3)):
    axes3[j].set_visible(False)

fig3.suptitle('Price Buy – Sample Daily Profiles', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig('output/fig_price_daily.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: output/fig_price_daily.png')

# PLOT 4 – Grid Buy / Battery Charge / Discharge


SYNC_SIGNALS = {
    'grid_buy_kW':          ('#C62828', 'Grid Buy'),
    'battery_charge_kW':    ('#1565C0', 'Battery Charge'),
    'battery_discharge_kW': ('#2E7D32', 'Battery Discharge'),
}

MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig4, axes4 = plt.subplots(4, 3, figsize=(15, 13), sharey=True)
axes4 = axes4.flatten()

for idx, month in enumerate(MONTH_ORDER):
    ax = axes4[idx]
    s  = _day_start(month, SYNC_REP_DAY)

    for col, (color, label) in SYNC_SIGNALS.items():
        vals = df[col].iloc[s:s + 24].values
        ax.plot(hours_24, vals, color=color, linewidth=1.8, label=label)
        ax.fill_between(hours_24, vals, alpha=0.07, color=color)

        if vals.max() > 0.5:
            pk = int(np.argmax(vals))
            ax.axvline(pk, color=color, linestyle=':', linewidth=1.0, alpha=0.65)

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel('Hour of day', fontsize=8)
    ax.set_ylabel('Power (kW)', fontsize=8)
    ax.set_title(f'{month}  (Day {SYNC_REP_DAY})', fontsize=10, fontweight='bold', pad=5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.20)

    if idx == 0:
        ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

fig4.suptitle(
    'Grid Buy · Battery Charge · Battery Discharge',
     fontsize=12, fontweight='bold',
)
plt.tight_layout(rect=[0, 0.01, 1, 0.94])
plt.savefig('output/fig_grid_battery_sync.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: output/fig_grid_battery_sync.png')
