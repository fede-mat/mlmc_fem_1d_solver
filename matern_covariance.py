import numpy as np
from scipy.special import kn, gamma
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
for N in range(6):
    plt.plot(x, kn(N, x), label='$K_{}(x)$'.format(N))
plt.ylim(0, 10)
plt.legend()
plt.title(r'Modified Bessel function of the second kind $K_n(x)$')
plt.show()

# Funzione di covarianza Matérn vettorizzata rispetto alla distanza r
def matern_covariance(r, nu, sigma, k):
    # Evita la divisione per zero e l'indeterminazione di kn calcolando solo dove r > 0
    # Quando r = 0, il valore limite è sigma**2
    res = np.zeros_like(r)
    
    # Maschera per i punti dove la distanza è maggiore di zero
    mask = r > 0
    
    # Calcolo per r > 0
    rk = r[mask] * k
    numerator = (sigma**2) * (rk**nu) * kn(nu, rk)
    denominator = (2**(nu - 1)) * gamma(nu)
    res[mask] = numerator / denominator
    
    # Caso r == 0
    res[~mask] = sigma**2
    return res

# Definizione della griglia spaziale (distanze dall'origine o coordinate dx, dy)
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)

# Calcola la distanza euclidea di ogni punto dall'origine (0,0)
R = np.sqrt(X**2 + Y**2)

# Calcolo dei valori Z
Z = matern_covariance(R, nu=2.5, sigma=4.0, k=10.0)  

# Creazione del grafico 3D
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

# Disegno della superficie
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)

# Configurazione dettagli estetici
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Covariance')
ax.set_title('Matérn Covariance')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Covariance')

# Ottimizzazione prospettiva iniziale (opzionale)
ax.view_init(elev=30, azim=45)

plt.show()