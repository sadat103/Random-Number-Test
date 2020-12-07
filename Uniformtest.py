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
alpha = float(input  ("Enter value of alpha :"))
U = np.zeros(N)
Z = np.zeros(N)
Z[0] = 1505103
U[0] = 1505103/(2 ** 31)
    #print(U)
for i in range(1,N):
    Z[i] = (65539 * Z[i-1]) % (2 ** 31)
    U[i] = Z[i]/(2 ** 31)
#print(Z)
uniform_numbers = np.round_(U,6)
print("Sub interval")
np.savetxt("Z.txt", Z, fmt="%s")
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")
print(K)
print("item per interval")
t= 1/K
sum1 = 0
k =0 
for i in np.arange(0,1,t):
    k = k+1
    x = np.where(np.logical_and(uniform_numbers >i , uniform_numbers <= i+t))
    print("Total number in %dth interval is %d "%(k,len(uniform_numbers[x])))
    sum=0
    for e in range(0,len(uniform_numbers[x])):
        sum = sum + ((K/N) * (uniform_numbers[x][e]-N/K)**2)
    print(" From %f to %f interval "%(i+0.01,i+t))
    print("Numbers are  =", uniform_numbers[x])
    print(sum)
    print("-------------------------------------------------------------------------------")
    sum1 = sum + sum1

print(" CHi Square is = " , sum1)
a = stats.chi2.ppf(q=alpha,df=k-1)
print(a)
if(sum1 > a):
    print("Reject")
else:
    print("Accept")

'''
Code to generate chi-square values where q are the levels and df is the degrees of freedom.
'''
'''
q_list = [0.25,0.5,0.75,0.9,0.95,0.975,0.99]

for df in range(1,20):
    st = ""
    for q in q_list:
        a = stats.chi2.ppf(q=q,df=df)
        st = st + str(a) + " "

    print(st)

'''
'''
Code to generate upper q critical point of the normal distribution
'''
#print(stats.norm.ppf(q=0.95))