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
    #print(U)
for i in range(1,500):
    Z[i] = (65539 * Z[i-1]) % (2 ** 31)
    U[i] = Z[i]/(2 ** 31)
#print(Z)
uniform_numbers = np.round_(U,6)
print("Sub interval")
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")
np.savetxt("Z.txt", Z, fmt="%s")
array = np.arange(36).reshape(6,6)
A=np.array([[ 4529.4,9044.9,  13568,  18091, 22615,27892],
       [ 9044.9,18097,27139,36187,45234,55789],
       [13568,27139,40721,54281,67852,83685],
       [18091,36187,5281,72414,90470,111580],
       [22615,45234,67852,90470,113262,139476],
       [27892,55789,83685,111580,139476,172860]])
print(A)

cnt = 1
runs = []
for i in range(1, len(uniform_numbers)):
    if uniform_numbers[i] >= uniform_numbers[i - 1]:
        cnt += 1
    else:
        runs.append(cnt)
        cnt = 1

# if the last few are in increasing order then they will be stored in the cnt
# else cnt will be 1
runs.append(cnt)
r = np.array(runs)
print(r)
sum=0
for i in range(0,len(r)):
    sum = sum + r[i]
print(sum)