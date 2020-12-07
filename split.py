import numpy as np

# arr = np.array([1, 2, 3, 4, 5, 6,2,3,1,5,4,7,8,9,10,11,12,13,14,15,2,3,4,5,0,1,2,7,8,6,4])
arr1 = np.array([0.86, 0.11, 0.23, 0.03, 0.13, 0.06, 0.55, 0.64, 0.87, 0.10])

cnt = 1
runs = []
for i in range(1, len(arr1)):
    if arr1[i] >= arr1[i - 1]:
        cnt += 1
    else:
        runs.append(cnt)
        cnt = 1

# if the last few are in increasing order then they will be stored in the cnt
# else cnt will be 1
runs.append(cnt)
r = np.array(runs)
print(r)
