import numpy as np

# Test with uniform mesh
# l=4
# xl=np.linspace(0,1,l+1)

# Test with non-uniform mesh
# xl=np.array([0,0.2,0.5,0.7,1])

def phi_0_hat(x):
    return 1-x

def phi_1_hat(x):
    return x

def phi_0_hat_prime(x):
    if len(x)==1:
        return -1
    else:
        return -np.ones_like(x)

def phi_1_hat_prime(x):
    if len(x)==1:
        return 1
    else:
        return np.ones_like(x)

def phi_i(x,a,b,c):
    if len(x)==1:
        if x<a or x>c:
            return 0
        elif a<=x<=b:
            return (x-a)/(b-a)
        else: # b<=x<=c:
            return (c-x)/(c-b)
    else:    
        return np.where((x<a) |(x>c),0,np.where((x<=b), (x-a)/(b-a), (c-x)/(c-b)))
    
def phi_i_prime(x,a,b,c):
    if len(x)==1:
        if x<a or x>c:
            return 0
        elif a<=x<=b:
            return 1/(b-a)
        else: # b<=x<=c:
            return -1/(c-b)
    else:
        return np.where((x<a) | (x>c), 0, np.where(x<=b, 1/(b-a), -1/(c-b)))     

def phi_i_0(x,a,b):
    if len(x)==1:
        if x<a or x>b:
            return 0
        else:
            return (b-x)/(b-a)
    else:
        return np.where((x<a) | (x>b), 0, (b-x)/(b-a))    

def phi_i_N(x,a,b):
    if len(x)==1:
        if x<a or x>b:
            return 0
        else:
            return (x-a)/(b-a)
    else:
        return np.where((x<a) | (x>b), 0, (x-a)/(b-a))  

def phi_i_0_prime(x,a,b):
    if len(x)==1:
        if x<a or x>b:
            return 0
        else:
            return -1/(b-a)
    else:
        return np.where((x<a) | (x>b), 0, -1/(b-a))

def phi_i_N_prime(x,a,b):
    if len(x)==1:
        if x<a or x>b:
            return 0
        else:
            return 1/(b-a)
    else:
        return np.where((x<a) | (x>b), 0, 1/(b-a))  

def rhs(f,xx,n_nodes=100):
    b = np.zeros_like(xx)
    for i in range(len(xx)):
        if i == 0:
            nodes = np.linspace(xx[i],xx[i+1],n_nodes)
            w = (xx[i+1]-xx[i])/(n_nodes-1)
            b[i] = w*np.sum(f(nodes)*phi_i_0(nodes,xx[i],xx[i+1]))
        elif i == len(xx)-1:
            nodes = np.linspace(xx[i-1],xx[i],n_nodes)  
            w = (xx[i]-xx[i-1])/(n_nodes-1)
            b[i] = w*np.sum(f(nodes)*phi_i_N(nodes,xx[i-1],xx[i]))
        else:
            nodes = np.linspace(xx[i-1],xx[i+1],2*n_nodes)
            w = (xx[i+1]-xx[i-1])/(2*n_nodes-1)
            b[i] = w*np.sum(f(nodes)*phi_i(nodes,xx[i-1],xx[i],xx[i+1]))
    return b

def local_mass_matrix(h):
    return h/6*np.array([[2,1],[1,2]])  

def local_stiffness_matrix(h):
    return 1/h*np.array([[1,-1],[-1,1]])

def local_bool_matrix(n,l):
    B=np.zeros((l+1,2))
    for i in range(2):
        B[i+n,i]=1
    return B.T

def global_mass_matrix(xl):
    l=len(xl)-1
    M=np.zeros((l+1,l+1))
    for i in range(l):
        h_i=xl[i+1]-xl[i]
        M+=local_bool_matrix(i,l).T@local_mass_matrix(h_i)@local_bool_matrix(i,l)
    return M

def global_stiffness_matrix(xl):
    l=len(xl)-1
    K=np.zeros((l+1,l+1))
    for i in range(l):
        h_i=xl[i+1]-xl[i]
        K+=local_bool_matrix(i,l).T@local_stiffness_matrix(h_i)@local_bool_matrix(i,l)
    return K

def weighted_stiffness_matrix(xl, w):
    l=len(xl)-1
    M=np.zeros((l+1,l+1))
    for i in range(l):
        h_i=xl[i+1]-xl[i]
        M+=local_bool_matrix(i,l).T@local_stiffness_matrix(h_i)@local_bool_matrix(i,l)*(w[i+1]+w[i])/2
    return M

#M=global_mass_matrix(xl)
#print(M)

#K=global_stiffness_matrix(xl)
#print(K)