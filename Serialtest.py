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
uniform_numbers = np.round_(U,6)
np.savetxt("Z.txt", Z, fmt="%s")
np.savetxt("uniform.txt", uniform_numbers, fmt="%s")

k =0

if d==2:
    N1 = np.zeros((int(N/d),d))

    for i in np.arange(0,N,d):
        k = k+1
        e = uniform_numbers[i:i+d]
        for j in range(0,len(e)):
             N1[k-1][j]= e[j]

    #print(N1)

    K_array = []
    a1 = math.sqrt(K**d)
    #print(a1)
    inter_val =1/K
    a2 = int(a1)
    N2 = np.zeros((a2,a2))
    myfile = open('Serial_number_range.txt', 'w')
    for i in range(0,len(N1)):
        for j in range(0,d):
            p = N1[i][j]
            k=0
            for m in np.arange(0,1,inter_val):
                k = k+1
                if np.logical_and(p >m , p <= m+inter_val):
                    #print("%dth tuple  %dth number %f is between ( %f to %f ) and interval is %d "%(i+1,j+1,p,m,m+inter_val,k))
                    myfile.write("%dth tuple  %dth number %f is between ( %.3f to %.3f ) and interval is %d\n "%(i+1,j+1,p,m,m+inter_val,k))
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

    print("Interval Count Array")
    print(N2)
    '''
    ll=0
    for i in range(0,len(N2)):
        for j in range(0,len(N2)):
            ll = ll + N2[i][j]
    print(ll)
    '''
    sum=0
    for i in range (0,len(N2)):
        for j in range(0,len(N2)):
            t = (N2[i][j] - N/(K**d))**2
            sum = sum + t

    CHi = ((K**d)/N)*sum
    print("Chi square is %f" %(CHi))
    a = stats.chi2.ppf(q=1-alpha,df=K**d -1)
    print("Chi ppf value is %f" %(a))
    if(CHi > a):
        print("Reject")
    else:
        print("Accept")


elif d==3:
    N1 = np.zeros((math.floor(N/d),d))

    for i in np.arange(0,N,d):
        k = k+1
        e = uniform_numbers[i:i+d]
        for j in range(0,len(e)):
             N1[k-2][j]= e[j]
    #print(N1)

    K_array = []
    inter_val =1/K
    myfile = open('Serial_number_range.txt', 'w')
    for i in range(0,len(N1)):
        for j in range(0,d):
            p = N1[i][j]
            k=0
            for m in np.arange(0,1,inter_val):
                k = k+1
                if np.logical_and(p >m , p <= m+inter_val):
                    #print("%dth tuple  %dth number %f is between ( %f to %f ) and interval is %d "%(i+1,j+1,p,m,m+inter_val,k))
                    myfile.write("%dth tuple  %dth number %f is between ( %.3f to %.3f ) and interval is %d\n "%(i+1,j+1,p,m,m+inter_val,k))
                    K_array.append(k)

    K_A = np.resize(np.array(K_array),(math.floor(N/d),d))
    np.savetxt("Serial_K_array.txt", K_A, fmt="%s")

    N2 = np.zeros((K,K,K))
    for i in range(0,len(K_A)):
        for j in range(0,d):
            if j==0:
                p = K_A[i][j]
            elif j==1:
                q = K_A[i][j]
            elif j==2:
                o = K_A[i][j]
        for d in range(0,len(N2)):
            for e in range(0,len(N2)):
                for f in range(0,len(N2)):
                    if (d+1)==p and (e+1)==q and (f+1)==o:
                         N2[d][e][f] = N2[d][e][f] + 1

    print("Interval Count Array")
    print(N2)
    '''
    ll=0
    for i in range(0,len(N2)):
        for j in range(0,len(N2)):
            for e in range(0,len(N2)):
                 ll = ll + N2[i][j][e]
    print(ll)
    '''
    sum=0
    for i in range (0,len(N2)):
        for j in range(0,len(N2)):
            for k in range(0,len(N2)):
                t = (N2[i][j][k] - N/(K**d))**2
                sum = sum + t

    CHi = ((K**d)/N)*sum
    print("Chi square is %f" %(CHi))
    a = stats.chi2.ppf(q=1-alpha,df=K**d -1)
    print("Chi ppf value is %f" %(a))
    if(CHi > a):
        print("Reject")
    else:
        print("Accept")