
import numpy as np
import math
import matplotlib.pyplot as plt

def target(w, x, y): #target function of log likehood method
    ret = 0
    n = len(x)
    for i in range(0,n):
        #print(x[i], w)
        rate = w.dot(x[i])
        #print(w, x[i], rate)
        ret += -rate*y[i] + math.log(1+math.exp(rate))
    return ret
pass
def test(w, testx, testy): #test and return accuracy
    n = len(testx)
    sum = 0
    for i in range(0,n):
        rate = w.dot(testx[i])
        #print("here",w, testx[i], rate)
        activation = 1./(math.exp(-rate)+1.)
        #print (rate)
        #print ("act", activation)
        if activation < 0.5 and testy[i] == 0 or (activation > 0.5 and testy[i] == 1):
           sum += 1
        elif activation == 0.5:
            sum += 0.5
    return sum/n
pass
def newton(weight, trainx, trainy):#Newton optimization of log likehood
    cur = 0
    n = len(trainx[0])
    while 1:
        old = cur
        #print(weight, trainx, trainy)
        cur = target(weight, trainx, trainy)
        #print(cur, old)
        if abs(old-cur) <= 0.000001:
            break
        dl = np.matrix(np.zeros(n)).astype(float)
        ddl = np.zeros((n,n)).astype(float)
        #print (n,dl, ddl)
        for i in range(0,len(trainx)):
            rate = weight.dot(trainx[i])
            #print(weight, trainx[i], weight.dot(trainx[i]))
            p = 1. - 1./(1.+math.exp(rate))
            #print("p=", p)
            #print(dl)
            dl += (-trainy[i]+p)*trainx[i]
            #print (i, np.matrix(trainx[i]).T, trainx[i], np.matrix(trainx[i]).T*trainx[i])
            ddl += np.matrix(trainx[i]).T*trainx[i]*p*(1-p)
        #print ("dl,ddl=",dl, np.linalg.pinv(ddl),dl*np.linalg.pinv(ddl))
        weight -= dl*np.linalg.pinv(ddl)
        #print(cur, weight)
    #print(test(weight, testx, testy))
    return weight
pass
def kFoldeCrossValidation(k, data, y):
    n = len(data[0])
    weight_final = np.zeros(n).astype(float)
    weight_final = np.matrix(weight_final)
    for it in range(0,k):
        weight = np.zeros(n).astype(float)#initialize train data and test data
        weight = np.matrix(weight)
        trainx = data[:math.floor(n*it/k)] + data[math.floor(n*(it+1)/k):]
        trainy = y[:math.floor(n*it/k)] + y[math.floor(n*(it+1)/k):]
        testx = data[math.floor(n*it/k):math.floor(n*(it+1)/k)]
        testy = y[math.floor(n*it/k):math.floor(n*(it+1)/k)]
        
        testx = np.array(testx).astype(float)#transfer data into array structure
        testy = np.array(testy).astype(int)
        trainx = np.array(trainx).astype(float)
        trainy = np.array(trainy).astype(int)
        #print(trainx, trainy)
        weight = newton(weight, trainx, trainy)

        weight_final += weight
    weight_final /= k
    data = np.array(data).astype(float)
    y = np.array(y).astype(int)
    print(test(weight_final, data, y))
    print(weight_final)
    return weight_final
pass
def Virtualization(data, y, weight_final): #Virtualization the scatter plot and draw the classification line,ONLY WORKS IN 2D
    x0,y0,x1,y1 = [],[],[],[]
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
    ax.set_title("Logic Regression of Melon Example")
    ax.set_xlabel("Density of melons")
    ax.set_ylabel("Sugar content")
    ax.scatter(x0,y0, label = 'Bad Melons', marker = 'o')#Plot scatter
    ax.scatter(x1,y1,  label = 'Good Melons', marker = 's')
    weight_final = np.array(weight_final)
    X = []
    Y = []
    for i in [1,9]:
        X.append(i*0.1)
        Y.append(-(i*0.1*weight_final[0][0] + weight_final[0][2])/weight_final[0][1])
    print (X,Y)
    ax.plot(X,Y, label = 'classification line')#Plot classification line
    ax.legend()
    plt.show() 

file = open("data-Iris.txt")
data = []
y = []
n = 0
while 1:#read data
    line = file.readline().strip()
    if not line:
        break
    pass
    line = line.split(',')
    k = len(line)
    #print (k, data, y)
    lines = []
    for i in range(1,k-1):
        lines.append(line[i])
    lines.append(1)
    data.append(lines)
    if line[-1] == '0':
        y.append(0)
    else:
        y.append(1)
    n += 1

#print(data, y)
weight_final = kFoldeCrossValidation(10, data, y)

weight_final = kFoldeCrossValidation(len(data), data, y)
#Virtualization(data, y, weight_final)