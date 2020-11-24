import numpy as np
import matplotlib.pyplot as plt


def knn(trainX, trainY, testX, K):
    dist = (((trainX - testX) ** 2).sum(1)) ** 0.5
    # (((trainX - testX) ** 2).sum(1)) ** 0.5  # np.sum(np.sqrt((trainX - testX) ** 2), axis=1)
    sortedDist = dist.argsort()
    classCount = {}
    for i in range(K):
        voteLabel = trainY[sortedDist[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxType = 0
    maxCount = -1
    for key, value in classCount.items():
        if value > maxCount:
            maxType = key
            maxCount = value
    return maxType


def getKValue(trainXSet, trainYSet, validXSet, validYSet, trainNum, validNum, batch_size):
    errorCount = 0.0
    acc = []
    k = []
    mTest = np.random.randint(0, validNum, trainNum // batch_size)
    for j in range(0, 20):
        k.append(j)  # trainNum // batch_size
        for i in range(len(mTest)):
            classifierResult = knn(trainXSet, trainYSet, validXSet[mTest[i]], j + 1)
            # print("KNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
            if (classifierResult != validYSet[mTest[i]]): errorCount += 1.0
        acc.append(((1 - errorCount / float(len(mTest))) * 100))
        errorCount = 0.0
        indexTmp = np.argwhere(acc == np.amax(acc))
        index = []
        for i in range(len(indexTmp)):
            index.append(indexTmp[i][0])
    plt.plot(k, acc)
    plt.title('Correct rate', fontsize=24)
    plt.xlabel('K', fontsize=14)
    plt.ylabel('Correct rate(%)', fontsize=14)
    plt.show()
    return int(np.mean(index))


def knnPredict(trainXSet, trainYSet, validXSet, validYSet, testX, testY, trainNum, validNum, batch_size):
    K = getKValue(trainXSet, trainYSet, validXSet, validYSet, trainNum, validNum, batch_size)
    print(K)
    errorCount = 0.0
    Num = len(testX)
    for i in range(Num):
        classifierResult = knn(trainXSet, trainYSet, testX[i], K)
        print("SVM得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
        if (classifierResult != testY[i]): errorCount += 1.0
    acc = (1 - errorCount / float(Num)) * 100
    print("\n辨识错误数量为: %d" % errorCount)
    print("\n辨识率为: %f ％" % acc)
    return acc
