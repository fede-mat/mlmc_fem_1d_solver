from auxiliary_fem import *
import matplotlib.pyplot as plt

# With this Test we assess the correctness of the rhs and global mass matrix functions by plotting the solution of M u = b, 
# where b is the rhs of a known function. We also plot the basis function phi_i_N to check its correctness.

def function(x):
    return np.sin(2 * np.pi * x)

#def function(x):
#    return np.ones_like(x)



# Use Chebyshev-Gauss-Lobatto nodes
#x_1 = np.linspace(0, 100, 101)
#x = - np.cos(x_1/100 *np.pi)

# Use uniform nodes
x = np.linspace(0, 1, 101)

plt.plot(x, function(x), label='f(x)')
plt.plot(x, phi_i_N(x, x[99],x[100]), label='phi_i_N(x, x[99], x[100])')
b = rhs(function, x, 100)
M = global_mass_matrix(x)
plt.plot(x, np.linalg.solve(M, b),'r--', label='rhs(f, x)')
plt.legend()
plt.show()
