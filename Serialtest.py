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
np.savetxt("Z.txt", Z, fmt="%s")
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")
#print(K)
print("item per interval")
#d= N/K
sum1 = 0
k =0
#print(uniform_numbers[1:4])
N1 = np.zeros((250,2))
print(N1)
for i in np.arange(0,500,2):
   k = k+1
   e = uniform_numbers[i:i+2]
   print("Total number in %dth tuple is %d "%(k,len(e)))
   print(uniform_numbers[i:i+2])
   for j in range(0,len(e)):
       N1[k-1][j]= e[j]
   print("-------------------------------------------------------------------------------")

print(N1)
print(N1[0][0]+1)
K_array = []
N2 = np.zeros((4,4))
for i in range(0,len(N1)):
    for j in range(0,2):
        p = N1[i][j]
        k=0
        for m in np.arange(0,1,0.25):
            k = k+1
            if np.logical_and(p >m , p <= m+0.25):
                print("%f  number in %f to %f interval and k and j is %d and %d "%(p,m,m+0.25,k,j))
                K_array.append(k)
               
K_A = np.resize(np.array(K_array),(250,2))
print(K_A)
for i in range(0,len(K_A)):
    for j in range(0,2):
        if j==0:
            p = K_A[i][j]
        else:
            q = K_A[i][j]
    for d in range(0,len(N2)):
        for e in range(0,len(N2)):
            if (d+1)==p and (e+1)==q:
                N2[d][e] = N2[d][e] + 1
print(N2)

