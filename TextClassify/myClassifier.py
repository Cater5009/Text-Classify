import numpy as np
import pandas as pd
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as tfidf
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
import math
from multiprocessing import Process


def loadDateSet(path):
    dataSet = pd.read_csv(path, header=None, delimiter=',', names=["index", "title", "description"])
    index = []
    title = []
    description = []
    t_and_d = []
    for row in dataSet.values:
        index.append(row[0])
        title.append(row[1])
        description.append(row[2])
        t_and_d.append(row[1] + row[2])
    return index, title, description, t_and_d, len(dataSet.values)


def Cal_P_R_F(target, realoutput):
    P = format(metrics.precision_score(target, realoutput, average='weighted'))
    R = format(metrics.recall_score(target, realoutput, average='weighted'))
    F = format(metrics.f1_score(target, realoutput, average='weighted'))
    print("  精度： ", '%.5f' % (float(P)))
    print("  召回： ", '%.5f' % (float(R)))
    print("  F1值： ", '%.5f' % (float(F)))


##计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  # 为所有可能分类创建字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # 以2为底数求对数
    return shannonEnt


# 依据特征划分数据集  axis代表第几个特征  value代表该特征所对应的值  返回的是划分后的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# # C4.5 选择最好的数据集(特征)划分方式  返回最佳特征下标
# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0]) - 1  # 特征个数
#     baseEntropy = calcShannonEnt(dataSet)
#     bestInfoGainrate = 0.0
#     bestFeature = -1
#     for i in range(numFeatures):  # 遍历特征 第i个
#         featureSet = set([example[i] for example in dataSet])  # 第i个特征取值集合
#         newEntropy = 0.0
#         splitinfo = 0.0
#         for value in featureSet:
#             subDataSet = splitDataSet(dataSet, i, value)
#             prob = len(subDataSet) / float(len(dataSet))
#             newEntropy += prob * calcShannonEnt(subDataSet)  # 该特征划分所对应的entropy
#             splitinfo -= prob * math.log(prob, 2)
#         if not splitinfo:
#             splitinfo = -0.99 * math.log(0.99, 2) - 0.01 * math.log(0.01, 2)
#         infoGain = baseEntropy - newEntropy
#         infoGainrate = float(infoGain) / float(splitinfo)
#         if infoGainrate > bestInfoGainrate:
#             bestInfoGainrate = infoGainrate
#             bestFeature = i
#     return bestFeature

# ID3 选择最好的数据集(特征)划分方式  返回最佳特征下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):   #遍历特征 第i个
        featureSet = set([example[i] for example in dataSet])   #第i个特征取值集合
        newEntropy= 0.0
        for value in featureSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #该特征划分所对应的entropy
        infoGain = baseEntropy - newEntropy

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 创建树的函数代码   python中用字典类型来存储树的结构 返回的结果是myTree-字典
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止继续划分  返回类标签-叶子节点
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有的特征时返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 得到的列表包含所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 多数表决的方法决定叶子节点的分类 ----  当所有的特征全部用完时仍属于多类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 排序函数 operator中的
                              #sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def store_tree(decesion_tree, filename):
    import pickle
    writer = open(filename, 'wb')
    pickle.dump(decesion_tree, writer)
    writer.close()


def read_tree(filename):
    import pickle
    reader = open(filename, 'rb')
    return pickle.load(reader)


def getCount(inputTree, dataSet, featLabels, count):
    # global num
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    # count=[]
    for key in secondDict.keys():
        rightcount = 0
        wrongcount = 0
        tempfeatLabels = featLabels[:]
        subDataSet = splitDataSet(dataSet, featIndex, key)
        tempfeatLabels.remove(firstStr)
        if type(secondDict[key]).__name__ == 'dict':
            getCount(secondDict[key], subDataSet, tempfeatLabels, count)
            # 在这里加上剪枝的代码，可以实现自底向上的悲观剪枝
        else:
            for eachdata in subDataSet:
                if str(eachdata[-1]) == str(secondDict[key]):
                    rightcount += 1
                else:
                    wrongcount += 1
            count.append([rightcount, wrongcount, secondDict[key]])
            # num+=rightcount+wrongcount


def cutBranch_uptodown(inputTree, dataSet, featLabels):  # 自顶向下剪枝
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            tempfeatLabels = featLabels[:]
            subDataSet = splitDataSet(dataSet, featIndex, key)
            tempfeatLabels.remove(firstStr)
            tempcount = []
            getCount(secondDict[key], subDataSet, tempfeatLabels, tempcount)
            print(tempcount)
            # 计算，并判断是否可以剪枝
            # 原误差率，显著因子取0.5
            tempnum = 0.0
            wrongnum = 0.0
            old = 0.0
            # 标准误差
            standwrong = 0.0
            for var in tempcount:
                tempnum += var[0] + var[1]
                wrongnum += var[1]
            old = float(wrongnum + 0.5 * len(tempcount)) / float(tempnum)
            standwrong = math.sqrt(tempnum * old * (1 - old))
            # 假如剪枝
            new = float(wrongnum + 0.5) / float(tempnum)
            if new <= old + standwrong and new >= old - standwrong:  # 要确定新叶子结点的类别
                ''' 
                #计算当前各个类别的数量多少，然后，多数类为新叶子结点的类别
            tempcount1=0
            tempcount2=0
            for var in subDataSet:
                if var[-1]=='0':
                tempcount1+=1
                else:
                tempcount2+=1
            if tempcount1>tempcount2:
                secondDict[key]='0'
            else:
                secondDict[key]='1'
                    '''
                # 误判率最低的叶子节点的类为新叶子结点的类
                # 在count的每一个列表类型的元素里再加一个标记类别的元素。
                wrongtemp = 1.0
                newtype = -1
                for var in tempcount:
                    if float(var[0] + var[1]) != 0:
                        if float(var[1] + 0.5) / float(var[0] + var[1]) < wrongtemp:
                            wrongtemp = float(var[1] + 0.5) / float(var[0] + var[1])
                            newtype = var[-1]
                secondDict[key] = str(newtype)


# 使用决策树执行分类
def ID3_Predict(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # index方法查找当前列表中第一个匹配firstStr变量的元素的索引
    classLabel = 0
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = ID3_Predict(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    if classLabel == 0:
        classLabel = np.random.randint(1, 5, 1)[0]
    return classLabel

def ID3_Predict_all(tree, matrix):
    index = []
    for row in matrix:
        index.append(ID3_Predict(tree, list(range(100)), row))
    return index

def myprocess(bagtrain, bagtest, targetindex, kernel, string):
    SVM = SVC(kernel=kernel)
    SVM.fit(bagtrain, TrainIndex)
    TestIndex_SvmOutput = SVM.predict(bagtest).tolist()
    Cal_P_R_F(TestIndex_SvmOutput, targetindex)


if __name__ == '__main__':
    print("读入训练数据...")
    TrainIndex, TrainTitle, TrainDescri, TrainWhole, TrainCount = loadDateSet('../ag_news_csv/train.csv')
    print("读入测试数据...")
    TestIndexTarget, TestTitle, TestDescri, TestWhole, TestCount = loadDateSet('../ag_news_csv/test.csv')

    print("生成TF-IDF词向量空间...")
    Title = TrainTitle + TestTitle  # --------------------------------------------------------------title合集
    Descri = TrainDescri + TestDescri  # -----------------------------------------------------------descri合集
    Whole = TrainWhole + TestWhole  # --------------------------------------------------------------Whole合集
    cvector = CountVectorizer(stop_words='english', min_df=2,
                              max_features=100)  # -----------------------------------特征提取器，避开英文停用词
    transformer = tfidf()  # -----------------------------------------------------------------------计算tfidf特征
    temp = cvector.fit_transform(Title)  # ------------------------------------------------------每一行是一篇文章的词向量
    # temp = transformer.fit_transform(cvector.fit_transform(Title))#------------------------------每一行是一篇文章的词向量
    TrainTitle_bag = temp[0:TrainCount]
    TestTitle_bag = temp[TrainCount:temp.shape[0]]
    temp = cvector.fit_transform(Descri)  # ------------------------------------------------------每一行是一篇文章的词向量
    # temp = transformer.fit_transform(cvector.fit_transform(Descri))#-----------------------------每一行是一篇文章的词向量
    TrainDescri_bag = temp[0:TrainCount]
    TestDescri_bag = temp[TrainCount:temp.shape[0]]
    temp = cvector.fit_transform(Whole)  # ------------------------------------------------------每一行是一篇文章的词向量
    # temp = transformer.fit_transform(cvector.fit_transform(Whole))# -----------------------------每一行是一篇文章的词向量
    TrainWhole_bag = temp[0:TrainCount]
    TestWhole_bag = temp[TrainCount:temp.shape[0]]
    # for row in TrainTitle_bag.toarray():
    #     print(row.indices)

    print("决策树分类...")
    # print(" ID3分类（使用Title）...")
    # TT_bag_id3 = TrainTitle_bag.toarray()
    # TT_bag_id3 = np.insert(TT_bag_id3, 100, values=TrainIndex, axis=1)
    # ID3 = createTree(TT_bag_id3.tolist(), list(range(100)))
    # store_tree(ID3, "ID3_traintitle.tree")
    # ID3 = read_tree("ID3_traintitle.tree")
    # # cutBranch_uptodown(ID3, TT_bag_id3.tolist(), list(range(100)))
    # TT_bag_id3 = np.insert(TestTitle_bag.toarray(), 100, values=TestIndexTarget, axis=1)
    # TestIndex_ID3Output = ID3_Predict_all(ID3, TT_bag_id3)
    # Cal_P_R_F(TestIndex_ID3Output, TestIndexTarget)
    print(" ID3分类（使用Description）...")
    TT_bag_id3 = TrainDescri_bag.toarray()
    TT_bag_id3 = np.insert(TT_bag_id3, 100, values=TrainIndex, axis=1)
    ID3 = createTree(TT_bag_id3.tolist(), list(range(100)))
    store_tree(ID3, "ID3_traindescri.tree")
    ID3 = read_tree("ID3_traindescri.tree")
    # cutBranch_uptodown(ID3, TT_bag_id3.tolist(), list(range(100)))
    TT_bag_id3 = np.insert(TestDescri_bag.toarray(), 100, values=TestIndexTarget, axis=1)
    TestIndex_ID3Output = ID3_Predict_all(ID3, TT_bag_id3)
    Cal_P_R_F(TestIndex_ID3Output, TestIndexTarget)
    print(" ID3分类（使用Title和Description）...")
    TT_bag_id3 = TrainWhole_bag.toarray()
    TT_bag_id3 = np.insert(TT_bag_id3, 100, values=TrainIndex, axis=1)
    ID3 = createTree(TT_bag_id3.tolist(), list(range(100)))
    store_tree(ID3, "ID3_trainwhole.tree")
    ID3 = read_tree("ID3_trainwhole.tree")
    # cutBranch_uptodown(ID3, TT_bag_id3.tolist(), list(range(100)))
    TT_bag_id3 = np.insert(TestWhole_bag.toarray(), 100, values=TestIndexTarget, axis=1)
    TestIndex_ID3Output = ID3_Predict_all(ID3, TT_bag_id3)
    Cal_P_R_F(TestIndex_ID3Output, TestIndexTarget)
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print(" sklearn自带Cart树分类（使用Title）...")
    # Cart = tree.DecisionTreeClassifier()
    # Cart.fit(TrainTitle_bag, TrainIndex)
    # TestIndex_CartOutput = Cart.predict(TestTitle_bag).tolist()
    # Cal_P_R_F(TestIndex_CartOutput, TestIndexTarget)
    # print(" sklearn自带Cart树分类（使用Description）...")
    # Cart = tree.DecisionTreeClassifier()
    # Cart.fit(TrainDescri_bag, TrainIndex)
    # TestIndex_CartOutput = Cart.predict(TestDescri_bag).tolist()
    # Cal_P_R_F(TestIndex_CartOutput, TestIndexTarget)
    # print(" sklearn自带Cart树分类（使用Title和Description）...")
    # Cart = tree.DecisionTreeClassifier()
    # Cart.fit(TrainWhole_bag, TrainIndex)
    # TestIndex_CartOutput = Cart.predict(TestWhole_bag).tolist()
    # Cal_P_R_F(TestIndex_CartOutput, TestIndexTarget)

    print("SVM分类...")
    # # linear
    # p1 = Process(target=myprocess, args=(TrainTitle_bag, TestTitle_bag, TestIndexTarget, "linear", " sklearn自带SVM分类（使用Title，核函数linear）..."))
    # p2 = Process(target=myprocess, args=(TrainDescri_bag, TestDescri_bag, TestIndexTarget, "linear", " sklearn自带SVM分类（使用Description，核函数linear）..."))
    # p3 = Process(target=myprocess, args=(TrainWhole_bag, TestWhole_bag, TestIndexTarget, "linear", " sklearn自带SVM分类（使用Title和Description，核函数linear）..."))
    # poly
    # p1 = Process(target=myprocess, args=(TrainTitle_bag, TestTitle_bag, TestIndexTarget, "poly", " sklearn自带SVM分类（使用Title，核函数poly）..."))
    # p2 = Process(target=myprocess, args=(TrainDescri_bag, TestDescri_bag, TestIndexTarget, "poly", " sklearn自带SVM分类（使用Description，核函数poly）..."))
    # p3 = Process(target=myprocess, args=(TrainWhole_bag, TestWhole_bag, TestIndexTarget, "poly", " sklearn自带SVM分类（使用Title和Description，核函数poly）..."))
    # #sigmoid
    # p1 = Process(target=myprocess, args=(TrainTitle_bag, TestTitle_bag, TestIndexTarget, "sigmoid", " sklearn自带SVM分类（使用Title，核函数sigmoid）..."))
    # p2 = Process(target=myprocess, args=(TrainDescri_bag, TestDescri_bag, TestIndexTarget, "sigmoid", " sklearn自带SVM分类（使用Description，核函数sigmoid）..."))
    # p3 = Process(target=myprocess, args=(TrainWhole_bag, TestWhole_bag, TestIndexTarget, "sigmoid", " sklearn自带SVM分类（使用Title和Description，核函数sigmoid）..."))
    # #rbf
    # p1 = Process(target=myprocess, args=(TrainTitle_bag, TestTitle_bag, TestIndexTarget, "rbf", " sklearn自带SVM分类（使用Title，核函数rbf）..."))
    # p2 = Process(target=myprocess, args=(TrainDescri_bag, TestDescri_bag, TestIndexTarget, "rbf", " sklearn自带SVM分类（使用Description，核函数rbf）..."))
    # p3 = Process(target=myprocess, args=(TrainWhole_bag, TestWhole_bag, TestIndexTarget, "rbf", " sklearn自带SVM分类（使用Title和Description，核函数rbf）..."))

    # p1.start()
    # p2.start()
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()

    print("Finish!\n")
