import sympy as sp

x = sp.symbols('x')

u_exact_sym = sp.sin(sp.pi/2 * x)
q_exact_sym = x + sp.sin(sp.pi * x)

exp_term = sp.exp(u_exact_sym)
dq_dx = sp.diff(q_exact_sym, x)

flux = exp_term * dq_dx

f_sym = -sp.diff(flux, x).simplify()

print(f_sym)