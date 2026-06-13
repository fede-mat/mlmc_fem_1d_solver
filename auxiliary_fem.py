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

#def finid_indices(alpha,beta,xx)
    # TO BE COMPLETED !!!
    return

def M_e_l_matrix(alpha,beta,xx):
    idx = np.where((xx[:-1] <= beta) & (xx[1:] >= alpha))[0]
    print(idx)
    k_1 = idx[0]
    k = idx[1]
    x_1 = xx[k_1]
    x = xx[k]
    M_e_l = np.zeros([2,2])
    # phi_i_e = ax + b
    a  =  - 1 / (x - x_1)
    b  =  x / (x - x_1)
    # phi_j_e = cx + d
    c  =  - a
    d = - x_1 / (x - x_1)

    # (M_e)_ij = int_ alpha to beta (ax+b)(cx + d) dx
    M_e_l[0,0] = (a*a*(beta**3-alpha**3))/3.0 + a*b*(beta**2-alpha**2) + b*b*(beta-alpha)
    M_e_l[1,0] = (a*c*(beta**3 - alpha**3))/3.0 + ((a*d + b*c)*(beta**2 - alpha**2))/2.0 + b*d*(beta - alpha)
    M_e_l[1,1] = (c*c*(beta**3-alpha**3))/3.0 + c*d*(beta**2-alpha**2) + d*d*(beta-alpha)
    M_e_l[0,1] = M_e_l[1,0]
    
    return M_e_l

def R_e_l_matrix(M_l,M_s):
    L_s = np.linalg.cholesky(M_s)
    L_l = np.linalg.cholesky(M_l)
    return np.linalg.inv(L_s).T @ L_l

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