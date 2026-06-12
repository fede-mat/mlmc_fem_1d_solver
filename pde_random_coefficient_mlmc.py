from auxiliary_fem import *
from random_field_generator import *
import time 

# DESCRIPTION:

def function_f(x):
    return np.ones_like(x)

def solve_pde_problem(x_G,x_D,f,g0,gN,k):
    u_sample = generate_random_field_(x_D, k,0,0)
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

def multilevel_montecarlo_estimation(num_samples_l,levels, x_G, x_D, f, g0, gN, k):
    norms = np.zeros(sum(num_samples_l))
    for l in range(levels):
        for i in range(num_samples_l[l]):
            _, q_norm_sample = solve_pde_problem(x_G, x_D, f, g0, gN, k)
            norms[i] = q_norm_sample
    return norms

seed = 2026
np.random.seed(seed) # for reproducibility

total_num_samples = 10000
percetage = np.array([0.17057356, 0.14457316, 0.10927159, 0.13943153 0.09303149 0.07291983
 0.08881164 0.0726671  0.05782876 0.05089134])
num_samples_l = percetage*total_num_samples
num_samples_l.round()
levels = np.linspace(1, 10, 10)

k=10
time_Start = time.time()
norms_1 = multilevel_montecarlo_estimation()
time_End = time.time()
sec = (time_End - time_Start)
norms_2 = ...
norms_3 = ...


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

mean_trend_1 = np.cumsum(norms_1) / np.arange(1, total_num_samples + 1)
mean_trend_2 = np.cumsum(norms_2) / np.arange(1, total_num_samples + 1)
mean_trend_3 = np.cumsum(norms_3) / np.arange(1, total_num_samples + 1)

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