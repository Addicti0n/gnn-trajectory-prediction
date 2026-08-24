import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(x, w=5):
    """κινητος μεσος ορος w epochs -> μειωνει τον θορυβο, κραταει την ταση"""
    return np.convolve(x, np.ones(w)/w, mode='valid')

# --- 1. Διαβασε το log (Tee-Object γραφει UTF-16) ---
raw = open('multiseed_results.txt', 'rb').read()
for enc in ['utf-16', 'utf-8', 'cp1252']:
    try:
        txt = raw.decode(enc)
        if 'lam=' in txt:
            break
    except Exception:
        pass

# --- 2. Παρσε τις per-epoch γραμμες ---
# μορφη:  lam=5 seed=2 ep=39: ADE=5.16 FDE=7.05 off=34.7%
pat = re.compile(r'lam=(\S+) seed=(\d+) ep=\s*(\d+): ADE=([\d.]+) FDE=([\d.]+) off=([\d.]+)%')
data = {}                                   # data[lambda][metric][seed] = [ανα epoch]
for line in txt.splitlines():
    m = pat.search(line)
    if not m:
        continue
    L, s = float(m.group(1)), int(m.group(2))
    d = data.setdefault(L, {'ade': {}, 'off': {}})
    d['ade'].setdefault(s, []).append(float(m.group(4)))
    d['off'].setdefault(s, []).append(float(m.group(6)))

# --- 3. Σχεδιασε: 2 panels (off-road & ADE) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {0.0: 'tab:red', 5.0: 'tab:blue'}
labels = {0.0: 'Baseline (λ=0)', 5.0: 'Map-Aware (λ=5)'}

for ax, key, title, ylab in [(axes[0], 'off', 'Off-Road Rate', 'off-road (%)'),
                             (axes[1], 'ade', 'Accuracy', 'minADE (m)')]:
    for L in sorted(data):
        arr = np.array([data[L][key][s] for s in sorted(data[L][key])])   # [seeds, epochs]
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        mu_s, sd_s = smooth(mu), smooth(sd)
        ep = np.arange(len(mu_s)) + 2                    # offset λογω 'valid'
        ax.plot(ep, mu_s, color=colors[L], label=labels[L], linewidth=2)
        ax.fill_between(ep, mu_s-sd_s, mu_s+sd_s, color=colors[L], alpha=0.2)
        ax.plot(np.arange(len(mu)), mu, color=colors[L], alpha=0.25, linewidth=0.8)  # raw, ξεθωριασμενο
 # ζωνη ±1 std
    ax.set_title(title); ax.set_xlabel('epoch'); ax.set_ylabel(ylab)
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Map-Aware Loss: επιδραση στην εκπαιδευση (3 seeds, ±1 std)')
plt.tight_layout()
plt.savefig('curves.png', dpi=120)
print('curves.png saved!')
