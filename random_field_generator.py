from auxiliary_fem import *
import matplotlib.pyplot as plt

# With this function we want to generate sample of a gaussian random field with Matern covariance using the result: 
#             (Id - k^-2 Laplacian)^ u = W, where W is a white noise.
#  We will use the FEM to solve this problem and generate samples of the random field.

# The weak form of the SPDE is:
#            (u,v) + k^-2 (grad u, grad v) = (W,v), for all v in H_0^1(0,1)
# We can discretize this problem using the FEM and solve the resulting linear system to get a sample of the random field.
# The discretized problem is:
#            (M + k^-2 K) u = b,
#  where M is the mass matrix and K is the stiffness matrix.
#  b ~ N(0,M) is a gaussian random vector with covariance matrix M 

def generate_random_field(xl, k,g0,gN):
    l = len(xl)-1
    M = global_mass_matrix(xl)
    K = global_stiffness_matrix(xl)
    A = M + k**(-2) * K
    b = np.zeros_like(xl)
    for i in range(l):
        h = xl[i+1] - xl[i]
        M_e = local_mass_matrix(h)
        b_e = np.random.multivariate_normal(mean=np.zeros(2), cov=M_e)
        b += local_bool_matrix(i, l).T  @ b_e
    # set boundary conditions to zero
    b_int = b[1:-1] - A[1:-1, 0] * g0 - A[1:-1, -1] * gN
    A_int = A[1:-1, 1:-1]
    u_int = np.linalg.solve(A_int, b_int)
    u = np.zeros(len(xl))
    u[0]    = g0
    u[-1]   = gN
    u[1:-1] = u_int
    return u

x = np.linspace(0,1,201)
k = 10
sample_num = 3
for i in range(sample_num):
    random_field_sample = generate_random_field(x, k,0,0)
    #plt.plot(x, random_field_sample)
    plt.plot(x, np.exp(random_field_sample))
plt.title("Sample of a Gaussian random field with Matern covariance")
plt.xlabel("x") 
plt.ylabel("u(x)")
plt.grid()
plt.show()
