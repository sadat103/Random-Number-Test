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
Z = np.zeros(500)
U = np.zeros(500)
Z[0] = 1505103
U[0] = 1505103/(2 ** 31)
    #print(U)
for i in range(1,500):
    Z[i] = (65539 * Z[i-1]) % (2 ** 31)
    U[i] = Z[i]/(2 ** 31)
#print(Z)
uniform_numbers = np.round_(U,6)
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")
np.savetxt("Z.txt", Z, fmt="%s")
array = np.arange(36).reshape(6,6)
A=np.array([[ 4529.4,9044.9,  13568,  18091, 22615,27892],
       [ 9044.9,18097,27139,36187,45234,55789],
       [13568,27139,40721,54281,67852,83685],
       [18091,36187,5281,72414,90470,111580],
       [22615,45234,67852,90470,113262,139476],
       [27892,55789,83685,111580,139476,172860]])
print("A matrix")
print(A)
array = np.arange(6)
B = np.array([1/6,5/24,11/120,19/720,29/5040,1/840])
print("B matrix")
print(B)
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
print("Run array")
print(r)
run_length_array = np.zeros(6)
for i in range(0,len(r)):
    if r[i] ==1:
        run_length_array[0]=run_length_array[0] +1
    elif r[i] ==2:
        run_length_array[1]=run_length_array[1] +1
    elif r[i] ==3:
        run_length_array[2]=run_length_array[2] +1
    elif r[i] ==4:
        run_length_array[3]=run_length_array[3] +1
    elif r[i] ==5:
        run_length_array[4]=run_length_array[4] +1
    else:
        run_length_array[5]=run_length_array[5] +1
print("Run length array")
print(run_length_array)

sum = 0 
for i in range(0,len(run_length_array)):
    for j in range(0,len(run_length_array)):
        p = A[i][j] * (run_length_array[i] - 500*B[i]) * (run_length_array[j]-500*B[j])
        sum = sum + p
R=sum/500
print(R)
a = stats.chi2.ppf(q=0.9,df=6)
print(a)
if(R > a):
    print("Reject")
else:
    print("Accept")

