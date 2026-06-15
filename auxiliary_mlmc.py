from auxiliary_fem import *
from random_field_generator import *
import time

# We want to assest the computational cost of each level of multilevel montecarlo
# and the variance of each sample. This will make us choose in a suitble way how 
# many samples to use in each level. 

seed = 2026
np.random.seed(seed)

levels = np.array([1,2,3,4,5,6,7,8,9,10])
levels = levels*10
print(levels)
k= 10
def function_f(x):
    return np.ones_like(x)

def solve_pde_problem(x_G,x_D,f,g0,gN,k):
    u_sample = generate_random_field_mc(x_D, k,0,0)
    u_sample_trunc = u_sample[(x_D >= -0.5) & (x_D <= 0.5)]
    weights = np.exp(u_sample_trunc)
    A = weighted_stiffness_matrix(x_G, weights)
    b = rhs(f,x_G,100)
    b_int = b[1:-1] - A[1:-1, 0] * g0 - A[1:-1, -1] * gN
    A_int = A[1:-1, 1:-1]
    q_int = np.linalg.solve(A_int, b_int)
    q = np.zeros(len(x_G))
    q[0]    = g0
    q[-1]   = gN
    q[1:-1] = q_int
    q_norm = np.sqrt(q.T @ global_mass_matrix(x_G) @ q) 
    return q , q_norm

def solve_coupled_pde_problem(x_Gl,x_Gl_1,x_Dl,x_Dl_1,X_Ds,f,g0,gN,k):

    u_sample_l ,u_sample_l_1 = generate_random_field_mlmc(x_Dl,x_Dl_1,X_Ds, k,0,0)

    u_sample_trunc_l = u_sample_l[(x_Dl >= -0.5) & (x_Dl <= 0.5)]
    u_sample_trunc_l_1 = u_sample_l_1[(x_Dl_1>=-0.5) & (x_Dl_1 <= 0.5)]

    weights_l = np.exp(u_sample_trunc_l)
    weights_l_1 = np.exp(u_sample_trunc_l_1)

    A_l = weighted_stiffness_matrix(x_Gl, weights_l)
    Al_1 = weighted_stiffness_matrix(x_Gl_1,weights_l_1)

    b_l = rhs(f,x_Gl,100)
    bl_1 = rhs(f,x_Gl_1,100)

    b_int_l = b_l[1:-1] - A_l[1:-1, 0] * g0 - A_l[1:-1, -1] * gN
    b_int_l_1 = bl_1[1:-1] - Al_1[1:-1, 0] * g0 - Al_1[1:-1, -1] * gN

    A_int_l= A_l[1:-1, 1:-1]
    A_int_l_1 = Al_1[1:-1,1:-1]

    q_int_l = np.linalg.solve(A_int_l, b_int_l)
    q_int_l_1 = np.linalg.solve(A_int_l_1,b_int_l_1)

    q_l = np.zeros(len(x_Gl))
    ql_1 = np.zeros(len(x_Gl_1))

    q_l[0]    = g0
    ql_1[0]   = g0
    q_l[-1]   = gN
    ql_1[-1]  = gN 

    q_l[1:-1] = q_int_l
    ql_1[1:-1] = q_int_l_1

    q_norm_l = np.sqrt(q_l.T @ global_mass_matrix(x_Gl) @ q_l)
    q_norm_l_1 = np.sqrt(ql_1.T @ global_mass_matrix(x_Gl_1)@ ql_1) 

    return q_l, ql_1, q_norm_l, q_norm_l_1

times = np.zeros_like(levels,dtype =float)
var = np.zeros_like(levels,dtype=float)
for j in range(len(levels)):
    if j==0:
        l = levels[j]
        x_G = np.linspace(-0.5,0.5,l+1)
        x_D = np.linspace(-1,1,2*l+1)
        aux_times = np.zeros(1000)
        aux_norm = np.zeros(1000)
        for i in range(1000):
            start = time.time()
            _ , estimated_norm  = solve_pde_problem(x_G,x_D,function_f,0,0,k)
            stop  = time.time()
            aux_times[i] = stop-start
            aux_norm[i] = estimated_norm
        times[j] = aux_times.mean()
        var[j] = aux_norm.var()
        print(f'level {l} completed: mean time {times[j]} and variance {var[j]}')
    else:
        l = levels[j]
        l_1 = levels[j-1]
        x_Gl = np.linspace(-0.5,0.5,l+1)
        x_Dl = np.linspace(-1,1,2*l+1)
        x_Dl_1 = np.linspace(-1,1,2*l_1+1)
        x_Gl_1 = np.linspace(-0.5,0.5,l_1)
        x_Ds = np.unique(np.concatenate((x_Dl,x_Dl_1)))
        x_Ds.sort()

        aux_times = np.zeros(1000)
        aux_norm = np.zeros(1000)
        for i in range(1000):
            start = time.time()
            _ , _, aux_l, auxl_1  = solve_coupled_pde_problem(x_Gl,x_Gl_1,x_Dl,x_Dl_1,x_Ds,function_f,0,0,k)
            stop  = time.time()
            aux_times[i] = stop-start
            aux_norm[i] = aux_l - auxl_1
        times[j] = aux_times.mean()
        var[j] = aux_norm.var()
        print(f'level {l} completed: mean time {times[j]} and variance {var[j]}')    

# Assuming that cost is directly proportional to time
M_l = (var/times)**0.5
M_l = M_l / sum(M_l)
print('Percetnage of sample for each level:',M_l)

with open('results/multilevel_monte_carlo_estimation_results.txt', 'a') as f:
    f.write(f'{M_l}')
    f.write('\n')