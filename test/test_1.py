from auxiliary_fem import *
import matplotlib.pyplot as plt
import sympy as sp
import time

# With this Test we test the correcteness of the FEM Library by solving the PDE: -u'' + u = f, with suitable boundary conditions. 
# We set u(x) and we compue the associated f(x). We check the correctness of the implementation by comparing 
# the numerical solution with the exact solution of the PDE. 
# We also plot a draf of the convergence of the error as we refine the mesh.



x = sp.symbols('x') 

# Automation of the computation of f(x) given u(x)
u_exact_sym = sp.sin(x**3 - x**2 + 0.25*x + 0.1) + 0.5
f_sym = -sp.diff(u_exact_sym, x, 2) + u_exact_sym

function_f_lambda = sp.lambdify(x, f_sym, 'numpy')
u_exact_lambda = sp.lambdify(x, u_exact_sym, 'numpy')

def function_f(x):
    return function_f_lambda(x)

def u_exact(x):
    return u_exact_lambda(x)

# Mesh generator
def make_mesh(n, perturb):
    x = np.linspace(0, 1, n+1)

    if perturb > 0:
        noise = perturb * (np.random.rand(n+1) - 0.5)
        x = x + noise
        x[0], x[-1] = 0, 1
        x = np.sort(x)

    return x
 
# Solver FEM
def solve_fem(xl,function,g0,gN):
    M = global_mass_matrix(xl)
    K = global_stiffness_matrix(xl)
    A = K + M
    b = rhs(function, xl)
    b_int = b[1:-1] - A[1:-1, 0] * g0 - A[1:-1, -1] * gN
    A_int = A[1:-1, 1:-1]
    u_int = np.linalg.solve(A_int, b_int)
    u = np.zeros(len(xl))
    u[0]    = g0
    u[-1]   = gN
    u[1:-1] = u_int

    return u

 
# Mesh Refinemente
meshes = [
    make_mesh(2,0.3),
    make_mesh(4,0.3),
    make_mesh(8,0.3),
    make_mesh(16,0.3),
    make_mesh(32,0.3),
    make_mesh(64,0.3),
    make_mesh(128,0.3) 
]

plt.figure(figsize=(8,5))

errors = np.zeros(len(meshes))

for i, xl in enumerate(meshes):
    tim_start = time.time()
    u = solve_fem(xl,function_f,u_exact(xl[0]),u_exact(xl[-1]))
    tim_end = time.time()
    print(f"Time taken for the test: {tim_end - tim_start:.5f} seconds with nodes={len(xl)}")
    plt.plot(xl, u, 'o-', label=f"nodes={len(xl)}, elements={len(xl)-1}")
    errors[i] = np.max(np.abs(u - u_exact(xl)))

plt.plot(xl, u_exact(xl), 'k--', label='Exact solution')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("FEM solution given different meshes")
plt.grid()
plt.legend()
plt.show()

plt.plot(np.log(errors[0:-2]/errors[1:-1]),'o-')
plt.title("Error Trend")
plt.show()

