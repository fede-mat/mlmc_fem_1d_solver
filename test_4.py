from auxiliary_fem import *

l = 5
x_l_1 = np.arange(0, 1,1/(l-1))
x_l = np.arange(0, 1,1/l)

M_l = global_mass_matrix(x_l)
M_l_1 = global_mass_matrix(x_l_1)

print("M_l: \n", M_l,"\n")

print("M_l_1: \n", M_l_1,"\n")