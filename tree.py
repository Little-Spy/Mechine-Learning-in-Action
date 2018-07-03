from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #计算总的数据项
    labelCounts = {}
    #为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  #不存在的话则出现次数为零
        labelCounts[currentLabel] += 1 #否则，若存在，将此值的键值对应+1
    #计算香农熵
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#划分数据集
def splitDataSet(dataSet, axis, value):  # 不甚理解  分别为待划分的数据集、
    # 划分数据集的特征、特征的返回值，axis可理解为以某位数据做特征
    retDataSet = []    # 创建一个新的列表对象，
    for featVec in dataSet:  # 数据集列表中的元素也是列表，遍历每个元素，将符合要求的值
        # 添加至新的列表中
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  #不包括第axis位的值
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):   # 数据要求：数据必须是一种由列表元素组成的列表
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 计算整个数据集的原始香农熵
    bestInfoGain = 0
    bestFeature  = -1
    for i in range(numFeatures):  # 遍历数据集中的所有特征
        featList = [example[i] for example in dataSet]  # 将数据集中所有的第i个特征值存入此列表中
        uniqueVals = set(featList)  # 快速得到列表中的所有不重复的元素值
        newEntropy = 0
        for value in uniqueVals:   # 计算每种数据集的新熵值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] =0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key =operator.itemgetter(1), reverse=True)
        # 字典.items(),以列表形式返回键-值对
    return sortedClassCount[0][0]


myDat, labels = createDataSet() # myDat是一个list，根据上文应为一个5行3列的list，
print(chooseBestFeatureToSplit(myDat))
print(myDat)
