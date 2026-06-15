import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auxiliary_fem import *
import matplotlib.pyplot as plt

l = 3
x_l_1 = np.linspace(0, 1, l)
x_l = np.linspace(0, 1,l+1)
# construct a supermesh by merging the two meshes and removing duplicates
x_s = np.unique(np.concatenate((x_l_1, x_l)))
x_s.sort()
print("Mesh 1:", x_l_1)
print("Mesh 2:", x_l)
print("Supermesh:", x_s)

def visualize_function_space(x, j):
    l = len(x)
    xx = np.linspace(x[0], x[-1], 100)
    
    # Controllo se l'indice è valido
    if j < 0 or j >= l:
        print(f"Errore: j deve essere tra 0 e {l-1}")
        return

    # Primo nodo (bordo sinistro)
    if j == 0:
        plt.plot(xx, phi_i_0(xx, x[0], x[1]), label=f"phi_{j}_{l}")
    # Ultimo nodo (bordo destro)
    elif j == l - 1:    
        plt.plot(xx, phi_i_N(xx, x[j-1], x[j]), label=f"phi_{j}_{l}")
    # Nodi interni
    else:
        plt.plot(xx, phi_i(xx, x[j-1], x[j], x[j+1]), label=f"phi_{j}_{l}")
        
    plt.title(f"FEM basis function phi_{j}")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


a,b = find_indices(x_s[1],x_s[2],x_l)
visualize_function_space(x_l,a)
R = R_e_l_matrix(x_s[1],x_s[2],x_l)
r_00 = R[0,0]
r_10 = R[1,0]
xx = np.linspace(0,1,100)
plt.plot(xx,r_00*phi_i(xx,x_l[a-1],x_l[a],x_l[b])+ r_10 * phi_i (xx,x_l[a],x_l[b],x_l[b+1]), '*')
plt.show()