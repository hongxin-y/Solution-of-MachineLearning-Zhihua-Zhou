
import numpy as np
import math
import matplotlib.pyplot as plt

def linearClass(x0,x1, k): #target function of log likehood method
    n0 = len(x0)
    n1 = len(x1)
    mu0 = sum(x0)/n0
    mu1 = sum(x1)/n1
    Sw = np.zeros((k,k)).astype(float)
    #print (mu0, mu1, Sw)
    for i in range(0,n0):#Calculate the inner-class scatter matrix
        diff = x0[i]-mu0
        Sw += diff.T*diff
    for i in range(0,n1):
        diff = x1[i]-mu1
        Sw += diff.T*diff
    #print("Sw=", Sw)
    return np.linalg.pinv(Sw)*np.matrix(mu0-mu1).T#return the theoretical solution of weight, which equals to Sw^(-1)*(mu0-mu1)
pass

def test(data, y, weight):
    mu0 = sum(data[y==0])/len(data[y==0])#Calculate the mean point of cluster
    mu1 = sum(data[y==1])/len(data[y==1])
    mu0 = np.matrix(mu0)*weight
    mu1 = np.matrix(mu1)*weight
    s = 0
    for i in range(len(data)):
        x = data[i]
        projection = np.matrix(x)*weight#Calculate the projection on the classification line
        if abs(projection-mu0) < abs(projection-mu1) and y[i]==0 or (abs(projection-mu1) < abs(projection-mu0) and y[i]==1):
            s += 1
        elif abs(projection-mu0) == abs(projection-mu1):
            s += 0.5
    return s/len(data)#return the accuracy

def Virtualization(data, y, weight_final): #Virtualization the scatter plot and draw the classification line,ONLY WORKS IN 2D
    x0,y0,x1,y1 = [],[],[],[]
    weight_final = weight_final.T
    data = np.array(data).astype(float)
    y = np.array(y).astype(int)#transfer data into numerical foramt
    for i in range(0,n):#classify two kinds of data
        if y[i] == 1:
            x0.append(data[i][0])
            y0.append(data[i][1])
        else:
            x1.append(data[i][0])
            y1.append(data[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    #print(x0, y0)
    ax.set_title("LDA of Melon Example")
    ax.set_xlabel("Density of melons")
    ax.set_ylabel("Sugar content")
    ax.scatter(x0,y0, label = 'Bad Melons', marker = 'o')#Plot scatter
    ax.scatter(x1,y1,  label = 'Good Melons', marker = 's')
    weight_final = np.array(weight_final)
    X = []
    Y = []
    for i in [1,3]:
        X.append(i*0.1)
        Y.append((i*0.1*weight_final[0][1])/weight_final[0][0]-0.8)
    #print (X,Y)
    ax.plot(X,Y, label = 'classification line')#Plot classification line
    ax.legend()
    plt.show() 

file = open("data.txt")
data = []
y = []
n = 0
while 1:#read data
    line = file.readline().split()
    if not line:
        break
    pass
    #line = line.split(',')
    k = len(line)
    #print (k, data, y)
    lines = []
    for i in range(1,k-1):
        lines.append(line[i])
    #lines.append(1)
    data.append(lines)
    if line[-1] == '0':
        y.append(0)
    else:
        y.append(1)
    n += 1

data = np.matrix(data).astype(float)
y = np.array(y).astype(float)#Trun data into matrix/array format
#print(data, y)
weight_final = linearClass(data[y==0],data[y==1], k-2)
print("accuracy=", test(data, y, weight_final))
Virtualization(data, y, weight_final)