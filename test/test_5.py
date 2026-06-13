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

def visualize_function_space(x):
    l = len(x)
    xx = np.linspace(x[0], x[len(x)-1], 100)
    for i in range(l):
        if i == 0:
            plt.plot(xx, phi_i_0(xx,x[i],x[i+1]), label=f"phi_0_{l}")
        elif i == len(x)-1:    
            plt.plot(xx, phi_i_N(xx,x[i-1],x[i]), label=f"phi_1_{l}")
        else:
            plt.plot(xx, phi_i(xx,x[i-1],x[i],x[i+1]), label=f"phi_{i}_{l}")
    plt.title("Piecewise linear basis functions")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    

#visualize_function_space(x_l_1)
#visualize_function_space(x_l)
#visualize_function_space(x_s)
visualize_function_space(np.linspace(-1,1,9))
visualize_function_space(np.linspace(-0.5,0.5,5))
plt.show()