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
    return [q, q_norm]

times = np.zeros_like(levels,dtype =float)
var = np.zeros_like(levels,dtype=float)
for j in range(len(levels)):
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

# Assuming that cost is directly proportional to time
M_l = (var/times)**0.5
M_l = M_l / sum(M_l)
print('Percetnage of sample for each level:',M_l)