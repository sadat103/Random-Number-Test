import numpy as np
theFile = open("Serial_K_array.txt", "r")
theInts = []
for val in theFile.read().split():
    theInts.append(float(val))
theFile.close()
print(theInts)
x = np.array(theInts)
print(x)
with open("write.txt","w") as filehandle:
    filehandle.writelines("%d , " % i for i in theInts )