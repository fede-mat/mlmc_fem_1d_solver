import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1000)
for N in range(6):
    plt.plot(x, kn(N, x), label='$K_{}(x)$'.format(N))
plt.ylim(0, 10)
plt.legend()
plt.title(r'Modified Bessel function of the second kind $K_n(x)$')
plt.show()

#plot of matern covariance function
def matern_covariance(x,y,nu,sigma,k):
    r=np.linalg.norm(x-y)
    if r==0:
        return 1
    else:
        return (sigma**2/(2**(nu-1)*np.math.gamma(nu)))*(r*k)**nu*kn(nu,r*k)
    
x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)
Z = matern_covariance(x,y,1,1,1)  
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Covariance')