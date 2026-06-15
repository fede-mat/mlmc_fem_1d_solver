import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary_fem import *

l = 100
xl = np.linspace(0,1,l+1)
xl_1 = np.linspace(0,1,l)
xs = np.unique(np.concatenate((xl,xl_1)))
xs.sort()

A = local_mass_matrix(xl[3]-xl[2])
B = M_e_l_matrix(xl[2],xl[3],xl)
# Questo ci dirà se la funzione M_e_l-matrix is correct, and yes it is !!!
print(A)
print(B)
print(A-B)

# Now we want to see if M_e_l on a super mesh element is really not SPD

C = M_e_l_matrix(xs[3],xs[4],xl_1)
print(C)
print(np.linalg.eigvals(C))
print(np.linalg.inv(C)@C)
