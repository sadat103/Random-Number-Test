from collections import Counter, defaultdict
import random
import numpy as np
import math
import multiprocessing as multi
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
#def random_number(n):
#N = int(input ("Enter value of n :")) 
#K = int(input  ("Enter value of k :"))
#alpha = float(input  ("Enter value of alpha :"))
U = np.zeros(500)
Z = np.zeros(500)
Z[0] = 1505103
U[0] = 1505103/(2 ** 31)
for i in range(1,500):
    Z[i] = (65539 * Z[i-1]) % (2 ** 31)
    U[i] = Z[i]/(2 ** 31)
uniform_numbers = np.round_(U,6)
np.savetxt("Z.txt", Z, fmt="%s")
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")
J = 2
h1 = 499/J
h = math.floor(h1)
print(h)

for k in range(0,h):
    x = uniform_numbers[1+k*J]
    y = uniform_numbers[1+(k+1)*J]
    print("x is %f" %(x))
    print("y is %f" %(y))