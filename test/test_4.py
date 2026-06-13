import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auxiliary_fem import *

l = 5
x_l_1 = np.arange(0, 1,1/(l-1))
x_l = np.arange(0, 1,1/l)

M_l = global_mass_matrix(x_l)
M_l_1 = global_mass_matrix(x_l_1)

L_l = np.linalg.cholesky(M_l)
L_l_1 = np.linalg.cholesky(M_l_1)


print("M_l: \n", M_l,"\n")
print("L_l: \n",L_l,"\n")
print("err: \n", M_l - L_l.T@L_l )



print("M_l_1: \n", M_l_1,"\n")
print("L_l_1: \n",L_l_1,"\n")