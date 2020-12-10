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
J = int(input  ("Enter value of J :"))
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
h1 = (N-1)/J
h = math.floor(h1)
print(h)
sum = 0
X =[]
Y =[]
for k in range(0,h):
    x = uniform_numbers[0+k*J] ## my case index start from 0 
    X.append(x)
    y = uniform_numbers[0+(k+1)*J]
    Y.append(y)
    d = x * y
    sum = sum + d

Correlation_Matrix = np.zeros((len(X),2))
for i in range(0,len(X)):
    for j in range(0,2):
        if j==0:
            Correlation_Matrix[i][j] = X[i]
        else:
            Correlation_Matrix[i][j] = Y[i]
            
#print("Correlation matrix")
#print(Correlation_Matrix)
np.savetxt("Correlation_Matrix.txt", Correlation_Matrix, fmt="%s")

print("Sum is %f" %(sum))

Ro_J = 12 * (sum/(h+1)) - 3
print("Ro_J is %f" %(Ro_J))

Var_Ro_J = (13*h + 7)/(h+1)**2
print("Var_Ro_J is %f" %(Var_Ro_J))

A_J = Ro_J/math.sqrt(Var_Ro_J)
print("A_J is %f" %(A_J))

t = 1 - alpha/2
Z_alpha = stats.norm.ppf(q=t)
print("Z_alpha is %f" %(Z_alpha))
if abs(A_J) > Z_alpha:
    print("Reject")
else:
    print("Accept")
