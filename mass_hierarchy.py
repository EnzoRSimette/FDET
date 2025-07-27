import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# Parâmetros do TFDE
R0 = 1e-32  # Planck scale
alpha = 0.63
n_values = np.arange(1, 21)  # Dimensional modes from 1 to 20

# Mass law: m_n ∝ n^α / R0
masses = (n_values**alpha) / R0

# Known particle masses for comparison (in GeV)
known_particles = {
    'Electron': 0.000511,
    'Muon': 0.105,
    'Tau': 1.776,
    'Top quark': 172.76,
    'W boson': 80.379
}

plt.figure(figsize=(10, 6), dpi=300)

# Plot TFDE mass hierarchy
plt.plot(n_values, masses, 'o-', lw=2, markersize=8, color='royalblue',
         label=r'$m_n = n^\alpha / R_0$ ($\alpha=0.63$)')

# Reference lines for known particles
colors = ['darkgreen', 'purple', 'crimson', 'darkorange', 'teal']
for i, (particle, mass) in enumerate(known_particles.items()):
    plt.axhline(y=mass*1e9, color=colors[i], linestyle='--', alpha=0.7)
    plt.text(20.5, mass*1e9, particle, va='center', ha='left', 
             fontsize=10, color=colors[i])

plt.yscale('log')
plt.title('Mass Hierarchy in TFDE Framework', fontsize=16)
plt.xlabel('Dimensional Index $n$', fontsize=14)
plt.ylabel('Mass (GeV)', fontsize=14)
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend(loc='upper left', frameon=False)
plt.xlim(0, 22)
plt.gca().yaxis.set_major_locator(LogLocator(base=100.0))

# Insert key equation
plt.annotate(r'$m_n \propto \frac{n^\alpha}{R_0}$', xy=(5, 1e20), 
             xytext=(5, 1e20), fontsize=18, color='navy')

plt.tight_layout()
plt.savefig('mass_hierarchy.pdf', bbox_inches='tight')