from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  #matrix.shape(0):输出矩阵的行数值
    normDataSet = dataSet - tile(minVals, (m, 1))  #tile(A, n),即为将数组A重复n次
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    # matrix.shape(0):输出矩阵的行数值;   matrix.shape(1):输出矩阵的列数值；
    #  shape(matrix):输出矩阵的行列值
    dataSetSize = shape(dataSet)[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #tile函数是将向量转变成矩阵，
    # 而这一步的含义是取得与每个点的误差
    sqDiffMat = diffMat ** 2
    #sum(axis=1):将矩阵的，每一行向量相加
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndices = distances.argsort()#排序，并返回索引值
    classCount = {} # 创建一个空字典
    for i in range(k):
        voteIlaber = labels[sortedDistIndices[i]]   #字典的键
        classCount[voteIlaber] = classCount.get(voteIlaber,0) + 1  #字典的键所对应的值
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)  # item会将字典的内容转换为list，查看P88，
    # 字典.items()返回一个键值对列表。    # 利用lambda来排序，这里记得要加上key
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  #得到文件行数
    returnMat = zeros((numberOfLines, 3)) #以零填充矩阵numpy
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  #截取掉所有的回车字符
        listFromLine = line.split('\t')   #使用\t将上一步得到的正行数据分割为一个元素列表
        returnMat[index, :] = listFromLine[0:3]  #提取前三个元素并存储至特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return  returnMat, classLabelVector


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent flier miles earned per year ?"))
    iceCream = float(input("liters of ice cream consumed per year ?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVials = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVials)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: " ,resultList[classifierResult - 1] )

print(classifyPerson())
