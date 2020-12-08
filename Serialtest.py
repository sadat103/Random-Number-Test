from collections import Counter, defaultdict
import random
import numpy as np
import math
import multiprocessing as multi
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
#def random_number(n):
N = int(input ("Enter value of n :")) 
K = int(input  ("Enter value of k :"))
d = int(input  ("Enter value of d :"))
alpha = float(input  ("Enter value of alpha :"))
U = np.zeros(N)
Z = np.zeros(N)
Z[0] = 1505103
U[0] = 1505103/(2 ** 31)
for i in range(1,N):
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

k =0
#print(uniform_numbers[1:4])
N1 = np.zeros((int(N/d),d))
print(N1)
for i in np.arange(0,N,d):
   k = k+1
   e = uniform_numbers[i:i+d]
   for j in range(0,len(e)):
       N1[k-1][j]= e[j]
inter_val =1/K
print(N1) 
K_array = []
a1 = math.sqrt(K**d)
print(a1)
a2 = int(a1)
N2 = np.zeros((a2,a2))
for i in range(0,len(N1)):
    for j in range(0,d):
        p = N1[i][j]
        k=0
        for m in np.arange(0,1,inter_val):
            k = k+1
            if np.logical_and(p >m , p <= m+inter_val):
                print("%dth tuple  %dth number %f is between ( %f to %f ) and interval is %d "%(i+1,j+1,p,m,m+inter_val,k))
                K_array.append(k)
               
K_A = np.resize(np.array(K_array),(int(N/d),d))
np.savetxt("Serial_K_array.txt", K_A, fmt="%s")
for i in range(0,len(K_A)):
    for j in range(0,d):
        if j==0:
            p = K_A[i][j]
        elif j==1:
            q = K_A[i][j]
    for d in range(0,len(N2)):
        for e in range(0,len(N2)):
            if (d+1)==p and (e+1)==q:
                N2[d][e] = N2[d][e] + 1
print(N2)
sum=0
for i in range (0,len(N2)):
    for j in range(0,len(N2)):
        t = (N2[i][j] - N/(K**d))**2
        sum = sum + t
        #print(sum)
CHi = ((K**d)/N)*sum
print("Chi square is %f" %(CHi))
a = stats.chi2.ppf(q=1-alpha,df=K**d -1)
print(a)
if(CHi > a):
    print("Reject")
else:
    print("Accept")

        