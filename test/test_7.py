from auxiliary_fem import *
l=4

xl_1 = np.linspace(0,1,l)
xl = np.linspace(0,1,l+1)

xs = np.unique(np.concatenate((xl_1,xl)))
xs.sort()

inx = np.where( ( xl == xs[1]) | (xl == xs[2]) )[0]

print(xl,"\n",xl_1,"\n",xs,"\n")
print(inx)
