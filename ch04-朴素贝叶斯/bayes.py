# coding: utf-8
from numpy import *

# 词表到向量的转换函数
def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]   # 1代表侮辱性言论，0代表正常言论
    return postingList, classVec


# 创建在所有文档中出现的去重后的词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)  # 并集
    return list(vocabSet)

# -------  词集模型(词集中每个单词只出现一次) -------
# 将词表转换为向量(表示词表中的单词在输入文档中是否出现)
def setOfWords2Vec(vocabList, inputSet):
    retVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:   # 将对应的词向量置1
            retVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in vacabulary." % word)
    return retVec


# -------  词袋模型(词袋中每个单词可能出现多次) -------
def bagOfWords2Vec(vocabList, inputSet):
    retVec = [0] *len(vocabList)
    for word in inputSet:
        if word in vocabList:   # 将对应的词向量+1
            retVec[vocabList.index(word)] += 1
    return retVec


# 朴素贝叶斯分类器训练函数
def trainNaiveBayes0(trainMatrix, trainCategory):
    numTrainDocs, numWords = shape(trainMatrix)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 防止概率相乘为 0, 把所有词出现次数初始化为 1, 总词数初始化为 2
    p0Nume = ones(numWords)
    p1Nume = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0   # 初始化概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   # 某个词语出现在侮辱性文档中
            p1Nume += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:   # 某个词出现在非侮辱性文档中
            p0Nume += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 防止太多小的浮点数相乘导致下溢出, 对乘积取自然对数
    p1Vect = log(p1Nume / p1Denom)
    p0Vect = log(p0Nume / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNaiveBayes(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    return 1 if p1 > p0 else 0


def testingNaiveBayes():
    listPosts, listClasses = loadDataSet()
    vocabList = createVocabList(listPosts)
    trainMat = []
    for doc in listPosts:
        trainMat.append(setOfWords2Vec(vocabList, doc))
    p0V, p1V, pAb = trainNaiveBayes0(array(trainMat), array(listClasses))

    # case 1
    testEntry = ['love', 'my', 'dalmaion']
    thisDoc = array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNaiveBayes(thisDoc, p0V, p1V, pAb))

    # case 2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNaiveBayes(thisDoc, p0V, p1V, pAb))


# ---------------- 使用朴素贝叶斯对电子邮件进行分类 ---------------------------
# 切分英语文本
def textParser(content):
    import re
    regex = re.compile(r'\W*')
    listOfTokens = regex.split(content)
    return [token.lower() for token in listOfTokens if len(token) > 2]


# 使用朴素贝叶斯进行电子邮件分类的测试
def testNaiveBayesToSpamEmail():
    emails = []
    emails_class = []
    for i in range(1, 26):
        # 垃圾邮件样本
        wordList = textParser(open('email/spam/%d.txt' % i).read())
        emails.append(wordList)
        emails_class.append(1)
        # 正常邮件样本
        wordList = textParser(open('email/ham/%d.txt' % i).read())
        emails.append(wordList)
        emails_class.append(0)
    
    vocabList = createVocabList(emails)
    trainingSet = list(range(50)) # 训练集
    testSet = []    # 测试集
    # 留存交叉验证: 随机选择数据一部分作为训练集, 剩余部分作为测试集
    # 随机构建训练集和测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for index in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, emails[index]))   # 使用词袋模型
        trainClasses.append(emails_class[index])
    
    p0V, p1V, pSpam = trainNaiveBayes0(array(trainMat), array(trainClasses))
    errorCount = 0
    for index in testSet:
        wordVect = bagOfWords2Vec(vocabList, emails[index])
        if classifyNaiveBayes(array(wordVect), p0V, p1V, pSpam) != emails_class[index]:
            errorCount += 1
    print('the error rate is: ', 1.0*errorCount/len(testSet))


# --------------------- 使用朴素贝叶斯分类器从个人广告中获取区域倾向 -------------------
# 计算词的出现频率
def calcMostFreq(vocabList, fullText, topN):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(
        freqDict.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    return sortedFreq[:topN]
    

def getLocalWords(feed1, feed0):
    summaryList = []
    summaries_class = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        # 第一个feed, 例子中为New York
        # 每次访问一条RSS源
        wordList = textParser(feed1['entries'][i]['summary'])
        summaryList.append(wordList)
        fullText.extend(wordList)
        summaries_class.append(1)
        # 第二个feed
        wordList = textParser(feed0['entries'][i]['summary'])
        summaryList.append(wordList)
        fullText.extend(wordList)
        summaries_class.append(0)
    vocabList = createVocabList(summaryList)

    # 停用词表: 语言中作为冗余/结构辅助性内容的词语表
    # 多语言停用词表例子 www.ranks.nl/resources/stopwords.html
    topN = 30
    topNWords = calcMostFreq(vocabList, fullText, topN)
    # 去除出现次数最多的N个词
    for word, _ in topNWords:
        if word in vocabList:
            vocabList.remove(word)
    # 随机生成训练集和测试集
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for index in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, summaryList[index]))
        trainClasses.append(summaries_class[index])
    p0V, p1V, pSpam = trainNaiveBayes0(array(trainMat), array(trainClasses))
    errorCount = 0
    for index in testSet:
        wordVect = bagOfWords2Vec(vocabList, summaryList[index])
        if classifyNaiveBayes(array(wordVect), p0V, p1V, pSpam) != summaries_class[index]:
            errorCount += 1
    print('the error rate is: ', 1.0*errorCount/len(testSet))
    return vocabList, p0V, p1V        


# 最具表征性的词汇显示函数
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = getLocalWords(ny, sf)
    THRESHOLD = -6.0
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > THRESHOLD:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > THRESHOLD:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(
        topSF,
        key=operator.itemgetter(1),
        reverse=True
    )
    print('-------  SF  -------')
    for item in sortedSF:
        print item[0]

    sortedNY = sorted(
        topNY,
        key=operator.itemgetter(1),
        reverse=True
    )
    print('-------  NY  -------')
    for item in sortedNY:
        print item[0]


if __name__ == '__main__':
    # testingNaiveBayes()
    # testNaiveBayesToSpamEmail()
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/search/stp?format=rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/search/stp?format=rss')
    vocabList, pSF, pNY = getLocalWords(ny, sf)
    getTopWords(ny, sf)

