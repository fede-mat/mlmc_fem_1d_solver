import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auxiliary_fem import *
import matplotlib.pyplot as plt
import sympy as sp

# With this test we aim to check if we are able to solve the pde:
# - (exp(u(x) q'(x)))' = f(x) for x in (0,1) and q(0)= q(1) = 0 or 1
# We choose q(x) as the exact solution and we compute the associated
# f(x) to check the correctness of our implementation. 
# To do so we replace the random coefficient exp(u(x)) with a dertministic function 
# with a similar bhavior of the original random field. The main goal is to assess the correctness 
# of the implementation of the weighted mass matrix. 

# Note: it works well for relativly smoth u(x), ( Ask Ullmann !!! ) 
# Comment: if the oscillation of the coefficient is too high we need to refine the mesh to get a 
# good approximation of the solution but in general this works quite well.

xx = np.linspace(0,1,101)
x = sp.symbols('x') 

# Automation of the computation of f(x) given q(x) and u(x)
u_exact_sym = sp.sin(sp.pi*x*10)
q_exact_sym = sp.sin(sp.pi*x*50) 
f_sym = -sp.diff(sp.exp(u_exact_sym)*sp.diff(q_exact_sym,x),x).simplify()

function_f_lambda = sp.lambdify(x, f_sym, 'numpy')
u_exact_lambda = sp.lambdify(x, u_exact_sym, 'numpy')
q_exact_lambda = sp.lambdify(x, q_exact_sym, 'numpy')

def function_f(x):
    return function_f_lambda(x)

def q_exact(x):
    return q_exact_lambda(x)

def u_exact(x):
    return u_exact_lambda(x)

def solve_deterministic_pde_problem(xl,f,g0,gN,weights):
    A = weighted_stiffness_matrix(xl, weights)
    b = rhs(f,xl,100)
    b_int = b[1:-1] - A[1:-1, 0] * g0 - A[1:-1, -1] * gN
    A_int = A[1:-1, 1:-1]
    q_int = np.linalg.solve(A_int, b_int)
    q = np.zeros(len(xl))
    q[0]    = g0
    q[-1]   = gN
    q[1:-1] = q_int
    q_norm = np.sqrt(q.T @ global_mass_matrix(xl) @ q) 
    return [q, q_norm]

[q, q_norm] = solve_deterministic_pde_problem(xx,function_f,q_exact(xx[0]),q_exact(xx[-1]),np.exp(u_exact(xx)))

plt.plot(xx, q_exact(xx), label='q_exact(x)')
plt.plot(xx, q,'r--', label='q_numerical(x)')
plt.title("Solution of the PDE with a deterministic coefficient")
plt.xlabel("x")
plt.ylabel("q(x)")
plt.legend()
plt.grid()
plt.show()

print(f"Norm of the numerical solution: {q_norm} \n Exact norm of the solution: {np.sqrt(q_exact(xx).T @ global_mass_matrix(xx) @ q_exact(xx))}")