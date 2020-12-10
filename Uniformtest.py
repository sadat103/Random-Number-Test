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
for i in range(1,N):
    Z[i] = (65539 * Z[i-1]) % (2 ** 31)
    U[i] = Z[i]/(2 ** 31)
#print(Z)
uniform_numbers = np.round_(U,6)
np.savetxt("Z.txt", Z, fmt="%s")
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")
t= 1/K
sum1 = 0
k =0
myfile = open('Uniformtest_number_range.txt', 'w')
for i in np.arange(0,1,t):
    k = k+1
    x = np.where(np.logical_and(uniform_numbers >i , uniform_numbers <= i+t))
    #print("Total number in %dth interval is %d "%(k,len(uniform_numbers[x])))
    d = ((K/N) * (len(uniform_numbers[x])-N/K)**2)
    sum1 = sum1 + d
    #print("From %f to %f interval "%(i,i+t))
    #print("Numbers are  =", uniform_numbers[x])
    myfile.write("Total number in %dth interval is %d\n"%(k,len(uniform_numbers[x])))
    myfile.write("From %f to %f interval Numbers are \n"%(i,i+t))
    myfile.write(str(uniform_numbers[x]))
    myfile.write("\nsum is = %f\n"%(d))
    myfile.write("-------------------------------------------------------------------------------\n\n")

print(" CHi Square is = " , sum1)
a = stats.chi2.ppf(q=1-alpha,df=k-1)
print(" CHi Square_ppf is = ", a)
if(sum1 > a):
    print("Reject")
else:
    print("Accept")
