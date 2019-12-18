import numpy as np
import pandas as pd
import random
import csv
from matplotlib import pyplot as plt

def Nmaxelements(list1, N):
    return np.argpartition(list1, N)[:N].tolist()

def errorcheck(arr1,arr2):
    return np.linalg.norm(np.subtract(np.asarray(arr1),np.asarray(arr2)))
ratings=pd.read_csv("output.csv").to_numpy()
sim=pd.read_csv("similarityEuc.csv").to_numpy()
asdf=[]
for i in range(3000):
    sum = 0
    num = 0
    for j in range(100):
        if(ratings[i][j]!=-1):
            sum+=ratings[i][j]
            num+=1
    asdf.append(sum/num)
asdf.sort()
print(max(asdf))
print(min(asdf))
plt.plot(asdf)
plt.show()

def weightedavg(list):
    sum=0
    total=0
    for i in list:
        if i[0]!=-1:
            sum+=1
            total+=i[0]
    return total/sum

def funct(x):
    predictlist = []
    uselist = []
    maxlist = Nmaxelements(sim[x].tolist(), 30)
    for y in range(100):
        if ratings[x][y] != -1:
            predictlist.append(ratings[x][y])
        else:
            for i in maxlist:
                if ratings[i][y] != -1 and sim[i][x] != 1:
                    uselist.append((ratings[i][y], sim[i][x]))
            predictlist.append(weightedavg(uselist))
    return predictlist

# x=0
# predictlist = []
# uselist = []
# maxlist=Nmaxelements(sim[x].tolist(),20)
# for y in range(100):
#     if ratings[x][y] == -1:
#         predictlist.append(-1)
#     else:
#         for i in maxlist:
#             if ratings[i][y] != -1 and sim[i][x]!=1:
#                 uselist.append((ratings[i][y],sim[i][x]))
#         predictlist.append(weightedavg(uselist))



# print(errorcheck(predictlist,ratings[0]))


def Rand(start, end, num):
    res = []

    for j in range(num):
        res.append(random.randint(start, end))

    return res

#print(errorcheck(Rand(0,10,100),ratings[0]))
# print(ratings.shape)
# with open('predictedProduct.csv','w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     for x in range(3000):
#         print(x)
#         writer.writerow(funct(x))

# countsto=[]
# for j in range(100):
#     count=0
#     for i in range(3000):
#         if ratings[i][j]==--1:
#             count+=1
#     countsto.append(count)
# print(countsto)
# print(sum(countsto))
