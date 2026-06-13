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

def generate_random_field_mc(xl, k,g0,gN):
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

def generate_random_field_mlmc(x_l,xl_1,x_s,k,g0,gN):
    l_1 = len(xl_1) - 1
    l = len(x_l) - 1
    s = len(x_s) - 1

    M_l = global_mass_matrix(x_l)
    Ml_1 = global_mass_matrix(xl_1)

    K_l = global_stiffness_matrix(x_l)
    Kl_1 = global_stiffness_matrix(xl_1)

    A_l = M_l + k**(-2)*K_l
    Al_1 = Ml_1 + k**(-2)*Kl_1

    b_l = np.zeros_like(x_l)
    bl_1= np.zeros_like(xl_1)
    b = np.concatenate((b_l,bl_1))
    b = b.astype(float)

    for i in range(s):
        h_s = x_s[i+1]-x_s[i]

        M_e_l = M_e_l_matrix(x_s[i],x_s[i+1],x_l)
        M_e_l_1 = M_e_l_matrix(x_s[i],x_s[i+1],xl_1)
        M_e_s = local_mass_matrix(h_s)

        R_e_l = R_e_l_matrix(M_e_l,M_e_s)
        R_e_l_1 = R_e_l_matrix(M_e_l_1,M_e_s)

        M_e_l_l_1 = R_e_l.T @ M_e_s @ M_e_l_1 

        M_e = np.array([M_e_l,M_e_l_l_1],[M_e_l_l_1.T, M_e_l_1])  # controllare come viene assemblata questa matrice!!!     
        
        b_e = np.random.multivariate_normal(mean=np.zeros(4),cov=M_e)
        b_e_l = b_e[0:2]
        b_el_1 = b_e[2:]

        indx_l = np.where((x_l >= x_s[i] ) & ( x_l <= x_s[i+1]))[0]
        indx_l_1 = np.where((xl_1 >= x_s[i] ) & ( xl_1 <= x_s[i+1]))[0]

        b_l += local_bool_matrix(indx_l[0],l).T @ b_e_l
        bl_1 += local_bool_matrix(indx_l_1[0],l_1).T @ b_el_1

    b_int_l = b[1:l_1] - A_l[1:-1, 0] * g0 - A_l[1:-1, -1] * gN
    b_int_l_1 = b[l_1+2:-1] - Al_1[1:-1, 0] * g0 - Al_1[1:-1, -1] * gN

    u_l = np.zeros(len(x_l))
    ul_1= np.zeros(len(xl_1))

    u_l[0] = g0
    u_l[-1] = gN
    ul_1[0] = g0
    ul_1[-1]= gN

    A_int_l = A_l[1:-1,1:-1]
    A_int_l_1 = Al_1[1:-1,1:-1]

    u_int_l = np.linalg.solve(A_int_l,b_int_l)
    u_int_l_1 = np.linalg.solve(A_int_l_1,b_int_l_1)

    u_l[1:-1]= u_int_l
    ul_1[1:-1]= u_int_l_1

    return np.concatenate((u_l,ul_1))

def visualize_mc_rf():
    x = np.linspace(0,1,201)
    k = 10
    sample_num = 3
    for i in range(sample_num):
        random_field_sample = generate_random_field_mc(x, k,0,0)
        #plt.plot(x, random_field_sample)
        plt.plot(x, np.exp(random_field_sample))
    plt.title("Sample of a Gaussian random field with Matern covariance")
    plt.xlabel("x") 
    plt.ylabel("u(x)")
    plt.grid()
    plt.show()
    return 

#visualize_mc_rf() 


