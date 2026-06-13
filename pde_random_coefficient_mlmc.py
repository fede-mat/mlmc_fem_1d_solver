from auxiliary_fem import *
from random_field_generator import *
import time 

# DESCRIPTION:
# Implementation of MLMC method to estrimate the expected value of the L^2 norm of the solution of 
# the follwoing pde with random coefficient:
# - (exp(u(x,w)) q'(x,w)))' = 1 for x in (-0.5,0.5) and w in Omega and q(-0.5,w)= q(0.5,w) = 0 
# Target: E[||q||_L^2(-0.5,0.5)] 

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

def solve_coupled_pde_problem(x_Gl,x_Gl_1,x_Dl,x_Dl_1,X_Ds,f,g0,gN,k):
    u_sample_l_l_1 = generate_random_field_mlmc(x_Dl,x_Dl_1,X_Ds, k,0,0)
    u_sample_l = u_sample_l_l_1[0:len(x_Dl)]
    u_sample_l_1 = u_sample_l_l_1[len(x_Dl)+1:]
    u_sample_trunc_l = u_sample_l[(x_Dl >= -0.5) & (x_Dl <= 0.5)]
    u_sample_trunc_l_1 = u_sample_l_1[(x_Dl_1>=-0.5)&(x_Dl_1 <= 0.5)]
    weights_l = np.exp(u_sample_trunc_l)
    weights_l_1 = np.exp(u_sample_l_1)
    A_l = weighted_stiffness_matrix(x_Gl, weights_l)
    Al_1 = weighted_stiffness_matrix(x_Gl_1,weights_l_1)
    b_l = rhs(f,x_Gl,100)
    bl_1 = rhs(f,x_Gl_1,100)
    b_int_l = b_l[1:-1] - A_l[1:-1, 0] * g0 - A_l[1:-1, -1] * gN
    b_int_l_1 = bl_1[1:-1]-Al_1[1:-1,0]*g0 -Al_1[1:-1,1]*gN
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
    return [np.concatenate((q_l,ql_1)),np.concatenate((q_norm_l,q_norm_l_1))]

def multilevel_montecarlo_estimation(num_samples_l,levels, f, g0, gN, k):
    total_num_samples = sum(num_samples_l)
    norms = np.zeros(total_num_samples.astype(int))
    for l in range(len(levels)):
        if l == 0:
            x_D = np.linspace(-1,1,2*levels[l].astype(int)+1)
            x_G = np.linspace(-0.5,0.5,levels[l].astype(int)+1)
            for i in range(num_samples_l[l].astype(int)):
                _ , norms[i] = solve_pde_problem(x_G,x_D,f,g0,gN,k)
        else: 
            x_Dl = np.linspace(-1,1,2*levels[l].astype(int)+1)
            x_Gl = np.linspace(-0.5,0.5,levels[l].astype(int)+1)
            x_Dl_1 = np.linspace(-1,1,2*levels[l-1].astype(int)+1)  
            x_Gl_1 = np.linspace(-0.5,0.5,2*levels[l-1].astype(int)+1) 
            x_Ds = np.unique(np.concatenate((x_Dl,x_Dl_1)))
            x_Ds.sort()    
            for i in range(num_samples_l[l].astype(int)):
                _ , aux = solve_coupled_pde_problem(x_Gl,x_Gl_1,x_Dl,x_Dl_1,x_Ds,f,g0,gN,k)
                norms[num_samples_l[l-1]+i] = aux[0]-aux[1]
    return norms

def visualize_mean_trend_from_norms(norms,num_samples_l,levels):
    mean_trend = np.zeros_like(norms,dtype=float)
    for i in range(len(levels)):
        if i==0:
            mean_trend[0:num_samples_l[i]] = np.cumsum(norms[0:num_samples_l[i]]) / np.arange(1,num_samples_l[i]+1)
        else:
            mean_trend[num_samples_l[i-1]:num_samples_l[i]] = np.cumsum(norms[num_samples_l[i-1]:num_samples_l[i]]) / np.arange(1,num_samples_l[i]+1)

    return np.cumsum(mean_trend)

seed = 2026
np.random.seed(seed) # for reproducibility

total_num_samples = 100
# We use auxiliary_mlmc.py to compute the optimal number of samples of each level
percetage = np.array([0.17057356, 0.14457316, 0.10927159, 0.13943153, 0.09303149, 0.07291983, 0.08881164, 0.0726671,  0.05782876, 0.05089134])
num_samples_l = percetage*total_num_samples
num_samples_l.round()
levels = np.linspace(1, 10, 10)
levels = levels*10
levels = levels.astype(int)

k=10
time_Start = time.time()
norms_1 = multilevel_montecarlo_estimation(num_samples_l,levels, function_f, 0,0, k)
time_End = time.time()
sec = (time_End - time_Start)
norms_2 = multilevel_montecarlo_estimation(num_samples_l,levels, function_f, 0,0, k)
norms_3 = multilevel_montecarlo_estimation(num_samples_l,levels, function_f, 0,0, k)

string = f"Estimated expected value of ||q||_L^2(-0.5,0.5) using Multilevel Monte Carlo with {total_num_samples} samples, seed {seed}: {np.mean(norms_1)}"
print(string)
string_1 = f"Time taken for Multilevel Monte Carlo estimation: {sec // 3600} hours {sec // 60 % 60:.0f} min {sec % 60:.2f} seconds"
print(string_1)

with open('results/multilevel_monte_carlo_estimation_results.txt', 'a') as f:
    f.write(string + '\n')
    f.write(string_1 + '\n')

plt.hist(norms_1, bins='auto', density=True, alpha=0.7, label='Multilevel Monte Carlo Estimates')
plt.xlabel('||q||_L^2(-0.5,0.5)')
plt.ylabel('Frequency')
plt.title('Distribution of ||q||_L^2(-0.5,0.5) Estimates')
plt.legend()
plt.grid()
#plt.autoscale(enable=True, axis='y', tight=True)
#plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(f'results/histogram_mlmc_estimates_seed_{seed}_samples_{total_num_samples}.png')
plt.show()

mean_trend_1 = visualize_mean_trend_from_norms(norms_1,num_samples_l,levels)
mean_trend_2 = visualize_mean_trend_from_norms(norms_2,num_samples_l,levels)
mean_trend_3 = visualize_mean_trend_from_norms(norms_3,num_samples_l,levels)

plt.plot(mean_trend_1, label=' Mean Trend 1')
plt.plot(mean_trend_2, label=' Mean Trend 2')
plt.plot(mean_trend_3, label=' Mean Trend 3')
plt.xlabel('Number of Samples')
plt.ylabel(' Mean')
plt.title('Convergence of Multilevel Monte Carlo Estimation')
plt.legend()
plt.grid()
#plt.autoscale(enable=True, axis='y', tight=True)
plt.savefig(f'results/convergence_mlmc_seed_{seed}_samples_{total_num_samples}.png')
plt.show()