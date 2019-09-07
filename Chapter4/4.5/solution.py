import numpy as np
import math
import matplotlib.pyplot as plt
import re
import regression
#To print Chinese
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

#Definition of tree nodes
class node:
    def __init__(self):
        self.children = []
        self.value = -1
        self.critical = None
        self.div = 0
        self.accuracy = -1
        self.weight = []
pass

def Entropy(dataset, y, k):
    if len(dataset) == 0:
        return 0
    p = [0 for i in range(k)]
    for i in range(len(dataset)):
        p[y[i]] += 1
    p = [p[i]/len(dataset) for i in range(k)]
    s = 0
    #print(p)
    for i in range(k):
        if p[i]!=0:
            s += p[i]*math.log2(p[i])
    return -s
pass

#this fuuntion return the gain list of all attributes
def Gain(dataset, label, y, k):
    gain = []
    entro = []
    for i in range(len(label)):
        entro.append([])
        for j in range(len(label[i])):
            entro[i].append([[],[]])

    #Construct continuous variable binary classification
    T = []
    th = 0
    for i in range(len(label)):
        if isinstance(label[i][0], float):
            if i>=1 and not isinstance(label[i-1][0], float):
                th = i
            t = []
            entro[i][0] = []
            entro[i][1] = []
            for j in range(len(label[i])-1):
                t.append((label[i][j]+label[i][j+1])/2)
            T.append(t)

    #Calculate the information entropy matrix
    for i in range(len(dataset)):
        for j in range(len(label)):
            if isinstance(label[j][0], float):
                for l in range(len(T[j-th])):
                    t = T[j-th][l]
                    entro[j][0].append([[],[]])
                    entro[j][1].append([[],[]])
                    #print(isinstance(data[i][j], float), isinstance(t,float))
                    #print(dataset[i][j])
                    if dataset[i][j] < t:
                        entro[j][0][l][0].append(dataset[i])
                        entro[j][0][l][1].append(y[i])
                    else:
                        entro[j][1][l][0].append(dataset[i])
                        entro[j][1][l][1].append(y[i])
            else:
                for d in range(len(label[j])):
                    #print(dataset[i], label[j])
                    if dataset[i][j] == label[j][d]:
                        entro[j][d][0].append(dataset[i])
                        entro[j][d][1].append(y[i])
    #print(entro)

    #Construct gain list, here we can use heap storing the gains to save time
    div = [None for _ in range(len(label))]
    for i in range(len(label)):
        e = Entropy(dataset, y, k)
        s = 0
        if isinstance(label[i][0], float):
            #Here can also be optimized by heap structure
            s = 0 
            for l in range(len(T[i-th])):
                #print(T[i-th][l], entro[i][0][l][0], entro[i][1][l][0])
                x1 = len(entro[i][0][l][1])/len(dataset)*Entropy(entro[i][0][l][0], entro[i][0][l][1], k)
                x2 = len(entro[i][1][l][1])/len(dataset)*Entropy(entro[i][1][l][0], entro[i][1][l][1], k)
                if s < e-x1-x2:
                    s = e-x1-x2
                    div[i] = T[i-th][l]
            gain.append(s)
        else:
            for j in range(len(label[i])):
                s += len(entro[i][j][1])/len(dataset)*Entropy(entro[i][j][0], entro[i][j][1], k)
            gain.append(e-s)
    ret = 0
    for i in range(len(gain)):
        if gain[i] > gain[ret]:
            ret = i
    #print (gain)
    return (ret,gain[ret], div)
pass

def Gini(dataset, y, k):
    if len(dataset) == 0:
        return 0
    gini = 0
    p = [0]*k
    for dy in y:
        p[dy] += 1
    for i in range(k):
        p[i]/=len(dataset)
        gini += p[i]*p[i]
    gini = 1-gini
    return gini
pass
def Gini_index(dataset, label, y, k):
    gini = []
    entro = []
    for i in range(len(label)):
        entro.append([])
        for j in range(len(label[i])):
            entro[i].append([[],[]])

    #Construct continuous variable binary classification
    T = []
    th = 0
    for i in range(len(label)):
        if isinstance(label[i][0], float):
            if i>=1 and not isinstance(label[i-1][0], float):
                th = i
            t = []
            entro[i][0] = []
            entro[i][1] = []
            for j in range(len(label[i])-1):
                t.append((label[i][j]+label[i][j+1])/2)
            T.append(t)

    #Calculate the information entropy matrix
    for i in range(len(dataset)):
        for j in range(len(label)):
            if isinstance(label[j][0], float):
                for l in range(len(T[j-th])):
                    t = T[j-th][l]
                    entro[j][0].append([[],[]])
                    entro[j][1].append([[],[]])
                    #print(isinstance(data[i][j], float), isinstance(t,float))
                    #print(dataset[i][j])
                    if dataset[i][j] < t:
                        entro[j][0][l][0].append(dataset[i])
                        entro[j][0][l][1].append(y[i])
                    else:
                        entro[j][1][l][0].append(dataset[i])
                        entro[j][1][l][1].append(y[i])
            else:
                for d in range(len(label[j])):
                    #print(dataset[i], label[j])
                    if dataset[i][j] == label[j][d]:
                        entro[j][d][0].append(dataset[i])
                        entro[j][d][1].append(y[i])
    #print(entro)

    #Construct gini index list, here we can use heap storing the gini indeces to save time
    div = [None for _ in range(len(label))]
    for i in range(len(label)):
        s = 0
        if isinstance(label[i][0], float):
            #Here can also be optimized by heap structure
            s = 0 
            for l in range(len(T[i-th])):
                #print(T[i-th][l], entro[i][0][l][0], entro[i][1][l][0])
                x1 = len(entro[i][0][l][1])/len(dataset)*Gini(entro[i][0][l][0], entro[i][0][l][1], k)
                x2 = len(entro[i][1][l][1])/len(dataset)*Gini(entro[i][1][l][0], entro[i][1][l][1], k)
                if s < x1+x2:
                    s = x1+x2
                    div[i] = T[i-th][l]
            gini.append(s)
        else:
            for j in range(len(label[i])):
                s += len(entro[i][j][1])/len(dataset)*Gini(entro[i][j][0], entro[i][j][1], k)
            gini.append(s)
    ret = 0
    for i in range(len(gini)):
        if gini[i] > gini[ret]:
            ret = i
    return (ret,gini[ret], div)

#Thit function return if the pre-cut should be done in such a devision, children is devision lists and valid is validation data.
def pre_cut(accuracy, children_x, children_y, valid, valid_y):
    #construct decisions in different sub-nodes
    #print("children", children_x, children_y)
    #print("valid",valid, valid_y)
    critical = []
    for child_y in children_y:
        num = {}
        for y in child_y:
            if y in num:
                num[y] += 1
            else:
                num[y] = 1
        mi = 0
        m = 0   
        for i,count in num.items():
            if count > m:
                mi = i
                m = count
        #print(mi, num, child_y)
        critical.append(mi)

    #Calculate the accuracy after partition
    acc_after = 0
    count = 0
    #print(critical)
    #print (valid, valid_y)
    for v in range(len(children_x)):
        for ty in valid_y[v]:
            if ty == critical[v]:
                acc_after += 1
            count += 1
    #print(count, acc_after)
    if count == 0:
        acc_after = 1
    else:
        acc_after /= count
    
    #print("before = ",accuracy,"after",acc_after)
    #print("pre-cut?", acc_after < accuracy)
    return acc_after <= accuracy

def post_cut(tree, data, y, label):
    acc_before = test(data, y, label, tree)
    for i,child in enumerate(tree.children):
        if child.value == -1:
            flag = True
            for c in child.children:
                if not flag:
                    break
                if c.value == -1:
                    flag = False
            if flag:
                acc_after = test(data, y, label, child)
                if acc_after>=acc_before:
                    new_node = node()
                    new_node.value = max(y)
                    tree.children[i] = new_node
            else:
                child = post_cut(child, data, y, label)
                tree.children[i] = child
    return tree


#This function return the root node of our decision tree
def DecisionTree(dataset, y, label, valid, validy, k, ifprecut):
    n = node()
    #One of recursion ending condition
    if len(set(y)) == 1:
        n.value = y[0]
        #print(n.value)
        return n
    
    #The codes below calculate the most frequency classification value and store it into 'mi'.
    num = {}
    for x in y:
        if x in num:
            num[x] += 1
        else:
            num[x] = 1
    m = 0
    mi = 0
    for i in range(len(num)):
        if num[i] >= m:
            m = num[i]
            mi = i

    #Two kinds of ending condition of recursion
    if len(label) == 0:
        n.value = mi
        #print(n.value)
        return n
    flag = True
    for i in range(len(dataset)):
        for j in range(i+1,len(dataset)):
            if dataset[i]!=dataset[j]:
                flag = False
    if flag:
        n.value = mi
        #print(n.value)
        return n
    
    #Calculate the thereold of continious and discrete variables
    th = len(dataset[0])
    for i in range(len(label)):
        if isinstance(label[i][0], float):
            if i>=1 and not isinstance(label[i-1][0], float):
                th = i
                break

    #Regression tree
    cont_data = []
    store_label = label[th:]
    for i in range(len(dataset)):
        cont_data.append(dataset[i][th:])
    #print(cont_data, th)
    w = regression.kFoldeCrossValidation(10, cont_data, y)
    w = np.array(w)
    #print("weight=",w)
    for i in range(len(dataset)):
        tmp = w.dot(cont_data[i])[0]
        dataset[i] = dataset[i][:th]
        dataset[i].append(tmp)
    attribute = []
    for data in dataset:
        attribute.append(data[-1])
    label = label[:th]
    label.append(attribute)
    #print(dataset, label)
    n.weight = w
    

    #Calculate the gain/gini_index and decide the attribute to branch next step
    index,g,div = Gain(dataset, label, y, k)
    n.critical = index
    n.div = div[index]
    #print(index, div)

    for i in range(len(dataset)):
        dataset[i] = dataset[i][:th] + cont_data[i]
    label = label[:th] + store_label

    #Continuous variables and Discrete variables have different judgement structure
    if isinstance(label[index][0], float):
        t = div[index]
        D = [[] for _ in range(2)]
        Dy = [[] for _ in range(2)]
        V = [[] for _ in range(2)]
        Vy = [[] for _ in range(2)]
        for i in range(len(dataset)):
            if dataset[i][index] < t:
                D[0].append(dataset[i][:index]+dataset[i][index+1:])
                Dy[0].append(y[i])
            else:
                D[1].append(dataset[i][:index]+dataset[i][index+1:])
                Dy[1].append(y[i])
        for i in range(len(valid)):
            if dataset[i][index] < t:
                V[0].append(valid[i][:index]+valid[i][index+1:])
                Vy[0].append(validy[i])
            else:
                V[1].append(valid[i][:index]+valid[i][index+1:])
                Vy[1].append(validy[i])
    else:
        D = [[] for _ in range(len(label[index]))]
        Dy = [[] for _ in range(len(label[index]))]
        V = [[] for _ in range(len(label[index]))]
        Vy = [[] for _ in range(len(label[index]))]
        for j in range(len(label[index])):
            for i in range(len(dataset)):
                if dataset[i][index] == label[index][j]:
                    D[j].append(dataset[i][:index]+dataset[i][index+1:])
                    Dy[j].append(y[i])
            for i in range(len(valid)):
                if valid[i][index] == label[index][j]:
                    V[j].append(valid[i][:index]+valid[i][index+1:])
                    Vy[j].append(validy[i])
    new_label = label[:index]+label[index+1:]

    #Calculate the accuracy of this node which will be used in deciding pre-cut.
    s = 0
    #print(validy, mi)
    for y in validy:
        if y == mi:
            s += 1
    if len(validy)!=0:
        s /= len(validy)
    else:
        s = 1
    n.accuracy = s
    #Do precut judgement
    #print ("D=",D, "V=",V)
    if ifprecut:
        if pre_cut(n.accuracy, D, Dy, V, Vy):
            #print("here", mi)
            n.value = mi
            return n

    #Recursion to visit childrens of the root
    #print(D)
    #print(Dy)
    #print(new_label)
    new_valid = []
    new_validy = []
    for v,vy in zip(V,Vy):
        new_valid += v
        new_validy += vy
    if isinstance(label[index][0], float):
        it = 2
    else:
        it = len(label[index])
    for v in range(it):
        if D[v]!=[]:
            #print(D[v], Dy[v])
            c = DecisionTree(D[v], Dy[v], new_label, V[v], Vy[v], k, ifprecut)
            n.children.append(c)
        else:
            c = node()
            c.value = mi
            n.children.append(c)
    #print(n.value)
    return n
pass

#This two functions are to test data and return accuracy rate
def test_one(data, y, label, root):
    if root.value != -1:
        #print (root.value)
        return y==root.value
    if isinstance(data[root.critical], float):
        new_label = label[:root.critical]+label[root.critical+1:]
        new_data = data[:root.critical]+data[root.critical+1:]
        if data[root.critical] < root.div:
            #print(label[root.critical])
            return test_one(new_data, y, new_label, root.children[0])
        else:
            #print(label[root.critical])
            return test_one(new_data, y, new_label, root.children[1])
    else:
        for i in range(len(root.children)):
            if label[root.critical][i] == data[root.critical]:
                new_label = label[:root.critical]+label[root.critical+1:]
                new_data = data[:root.critical]+data[root.critical+1:]
               #print(i, data[root.critical], root.critical, label[root.critical][i])
                return test_one(new_data, y, new_label, root.children[i])
    return False
pass
def test(dataset, y, label, root):
    s = 0
    for i in range(len(dataset)):
        judge = test_one(dataset[i], y[i], label, root)
        #print(judge)
        s += int(judge)
    if len(dataset) == 0:
        return 1
    return s/len(dataset)
pass

#Virtualization of tree, copied from https://blog.csdn.net/wancongconghao/article/details/71171981
decision_node = dict(boxstyle="sawtooth",fc="0.8")
leaf_node = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")
#获取树的叶子结点个数（确定图的宽度）
def get_leaf_num(tree):
    leaf_num = 0
    if len(tree.keys())==0:
        return 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__=="dict":
            leaf_num +=get_leaf_num(next_dict[key])
        else:
            leaf_num +=1
    return leaf_num
#获取数的深度（确定图的高度）
def get_tree_depth(tree):
    depth = 0
    if len(tree.keys())==0:
        return 0  
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            thisdepth = 1+ get_tree_depth(next_dict[key])
        else:
            thisdepth = 1
        if thisdepth>depth: depth = thisdepth
    return depth
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
#在父子节点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = get_leaf_num(myTree)
    depth = get_tree_depth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decision_node)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(get_leaf_num(inTree))
    plotTree.totalD = float(get_tree_depth(inTree))
    if plotTree.totalW == 0:
        print("No nodes in the tree")
        return
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

#Transfer tree into dictionary form
def tree_to_dic(tree, labels, l):
    dic = {}
    if tree.value==-1: #not leaf node
        label = labels[tree.critical]
        new_labels = labels[:tree.critical]+labels[tree.critical+1:]
        new_l = l[:tree.critical] + l[tree.critical+1:]
        subdic = {}
        if isinstance(l[tree.critical][0], float):
            w = np.array(tree.weight)[0]
            #print(w)
            if len(w) != 0:
                label = ""
                for i in range(len(w)):
                    label += str(w[i]) + labels[i+len(labels)-len(w)]
                    if i != len(w)-1:
                        label += '+\n'
                label += '\n<' + str(tree.div)
            else:
                label = label + "<" + str(tree.div)
        for i in range(len(tree.children)):
            child = tree.children[i]
            if isinstance(l[tree.critical][i], float):
                la = 'Yes' if i==0 else 'No'
            else:
                la = l[tree.critical][i]
            if child.value == -1:
                #print(new_l[child.critical])
                subdic[la] = tree_to_dic(child, new_labels, new_l)
            else:
                subdic[la] = '好瓜' if child.value==1 else '坏瓜'
        dic[label] = subdic
    #print (dic)
    return dic
pass

#start of main program
value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
file = open("train.txt")
train = []
train_y = []
n = 0
while 1:#read data
    line = file.readline().strip()
    if not line:
        break
    pass
    line = line.split(',')
    k = len(line)
    lines = []
    for i in range(1,k-1):
        lines.append(line[i])
    train.append(lines)
    if line[-1] == '否':
        train_y.append(0)
    else:
        train_y.append(1)
    n += 1
for i in range(len(train)):
    for j in range(len(train[i])):
        if value.match(train[i][j]):
            train[i][j] = float(train[i][j])
#print(train, train_y)
file = open("validation.txt")
validation = []
validation_y = []
n = 0
while 1:#read data
    line = file.readline().strip()
    if not line:
        break
    pass
    line = line.split(',')
    k = len(line)
    lines = []
    for i in range(1,k-1):
        lines.append(line[i])
    validation.append(lines)
    if line[-1] == '否':
        validation_y.append(0)
    else:
        validation_y.append(1)
    n += 1
for i in range(len(validation)):
    for j in range(len(validation[i])):
        if value.match(validation[i][j]):
            validation[i][j] = float(validation[i][j])
#print(validation, validation_y)

'''
#swap attributes to test different answers
for i,x in enumerate(train):
    x = [x[4]]+x[:4]+x[5:]
    train[i] = x
for i,x in enumerate(validation):
    x = [x[4]]+x[:4]+x[5:]
    validation[i] = x
train_y = [train_y[4]]+train_y[:4] + train_y[5:]
validation_y = [validation_y[4]] + validation_y[:4] + validation_y[5:]
print(train, train_y, validation, validation_y)
'''

#Create attribute set of every label
label = []
for i in range(0,len(train[0])):
    attribute = []
    for a in train:
        if not a[i] in attribute:
            attribute.append(a[i])
    label.append(sorted(attribute))

#print("here",train, validation)
#Create the decision tree without pre-cut nor post-cut
tree = DecisionTree(train, train_y, label, validation, validation_y, 2, False)

#test the decision tree (should be always 100% because I just use the generating data to test)
print(test(validation, validation_y, label, tree))

#Transfer tree into a dictionary form and draw the tree graph
labels = ['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
t = tree_to_dic(tree, labels, label)
print (t)
createPlot(t)

#Do the post-cut and draw the graph
tree = post_cut(tree, validation, validation_y, label)
print(test(validation, validation_y, label, tree))
t = tree_to_dic(tree, labels, label)
print(t)
createPlot(t)

#Do the pre-cut and draw the graph
#print(train, validation)
tree = DecisionTree(train, train_y, label, validation, validation_y, 2, True)
print(test(validation, validation_y, label, tree))
t = tree_to_dic(tree, labels, label)
print (t)
createPlot(t)

