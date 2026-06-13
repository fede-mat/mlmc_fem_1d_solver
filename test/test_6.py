import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

la = len(a)-1
lb = len(b)-1

c = np.concatenate((a,b))
print(c)
print(c[1:la],c[la+2:-1])
idx =  np.where((a >= 2) & (a <= 3))[0]
print(idx[0],'ciao',idx[1])