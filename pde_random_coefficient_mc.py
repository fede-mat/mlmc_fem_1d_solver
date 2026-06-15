from auxiliary_fem import *
from random_field_generator import *
import time 

# DESCRIPTION:
# We now want to estrimate the expected value of the L^2 norm of the solution of the follwoing pde with random coefficient:
# - (exp(u(x,w)) q'(x,w)))' = 1 for x in (-0.5,0.5) and w in Omega and q(-0.5,w)= q(0.5,w) = 0 
# Target: E[||q||_L^2(-0.5,0.5)] 

# QUESTIONS & TO DO:
#- fix / organize the functions and input parameters 
#- plot of the convegrence of MLMC and one solution of the PDE for a given sample of u(x,w)
#- ask Ullmann about the coefficient sigma, kappa ecc... to reproduce the results; take into account that: || u ||_L^2 ~ h^(-1/2) || u_h ||_L^2 and || u_h ||_L^2 = u.T @ M @ u 
#- verificare empiricamente che questo modo di fare sampling dei vettori gaussiani è effettivamente più efficiente (tic - toc)
#- chiedere a stephan lunowa se come rioganizzare queste funzioni in una libreria 

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

def monte_carlo_estimation(num_samples,x_G,X_D,f,g0,gN,k):
    norms = np.zeros(num_samples)
    for i in range(num_samples):
        _, q_norm_sample = solve_pde_problem(x_G,X_D,f,g0,gN,k)
        norms[i] = q_norm_sample
        if  i % (num_samples // 10) == 0:
            print(f"Monte Carlo estimation: {i}/{num_samples} samples completed in {time.time() - time_Start:.2f} seconds")
    return norms


seed = 2026
np.random.seed(seed) # for reproducibility

num_samples = 100
x_D = np.linspace(-1,1,201)
x_G = np.linspace(-0.5,0.5,101)
k=10
time_Start = time.time()
norms_1 = monte_carlo_estimation(num_samples,x_G,x_D,function_f,0,0,k)
time_End = time.time()
sec = (time_End - time_Start)
norms_2 = monte_carlo_estimation(num_samples,x_G,x_D,function_f,0,0,k)
norms_3 = monte_carlo_estimation(num_samples,x_G,x_D,function_f,0,0,k)

string = f"Estimated expected value of ||q||_L^2(-0.5,0.5) using Monte Carlo with {num_samples} samples, seed {seed}: {np.mean(norms_1)}"
print(string)
string_1 = f"Time taken for Monte Carlo estimation: {sec // 3600} hours {sec // 60 % 60:.0f} min {sec % 60:.2f} seconds"
print(string_1)

with open('results/monte_carlo_estimation_results.txt', 'a') as f:
    f.write(string + '\n')
    f.write(string_1 + '\n')

plt.hist(norms_1, bins='auto', density=True, alpha=0.7, label='Monte Carlo Estimates')
plt.xlabel('||q||_L^2(-0.5,0.5)')
plt.ylabel('Frequency')
plt.title('Distribution of ||q||_L^2(-0.5,0.5) Estimates')
plt.legend()
plt.grid()
#plt.autoscale(enable=True, axis='y', tight=True)
#plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig(f'results/histogram_mc_estimates_seed_{seed}_samples_{num_samples}.png')
plt.show()

mean_trend_1 = np.cumsum(norms_1) / np.arange(1, num_samples + 1)
mean_trend_2 = np.cumsum(norms_2) / np.arange(1, num_samples + 1)
mean_trend_3 = np.cumsum(norms_3) / np.arange(1, num_samples + 1)

plt.plot(mean_trend_1, label=' Mean Trend 1')
plt.plot(mean_trend_2, label=' Mean Trend 2')
plt.plot(mean_trend_3, label=' Mean Trend 3')
plt.xlabel('Number of Samples')
plt.ylabel(' Mean')
plt.title('Convergence of Monte Carlo Estimation')
plt.legend()
plt.grid()
#plt.autoscale(enable=True, axis='y', tight=True)
plt.savefig(f'results/convergence_mc_seed_{seed}_samples_{num_samples}.png')
plt.show()

[q, estimated_norm] = solve_pde_problem(x_G,x_D,function_f,0,0,k)
plt.plot(x_G, q)
plt.title("Solution of the PDE for a given sample of u(x,w)")
plt.xlabel("x")
plt.ylabel("q")
plt.grid()
plt.legend()
plt.savefig(f'results/solution_pde_sample.png')
plt.show()        