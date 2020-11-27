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
    print("start KNN")
    errorCount = 0.0
    acc = []
    k = []
    mTest = np.random.randint(0, validNum, 500)
    for j in range(0, 50):
        k.append(j)  # trainNum // batch_size
        for i in range(len(mTest)):
            classifierResult = knn(trainXSet[:2500], trainYSet[:2500], validXSet[mTest[i]], j + 1)
            # print("KNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
            if (classifierResult != validYSet[mTest[i]]): errorCount += 1.0
        acc.append(((1 - errorCount / float(len(mTest))) * 100))
        errorCount = 0.0
        indexTmp = np.argwhere(acc == np.amax(acc))
        index = []
        for i in range(len(indexTmp)):
            index.append(indexTmp[i][0])
    plt.plot(k, acc)
    plt.title('KNN Correct rate', fontsize=24)
    plt.xlabel('K', fontsize=14)
    plt.ylabel('Correct rate(%)', fontsize=14)
    plt.show()
    print("\nValid KNN辨识率为: %f ％" % np.mean(acc))
    print("finished KNN")
    return int(np.mean(index))


def knnPredict(trainXSet, trainYSet, validXSet, validYSet, testX, testY, trainNum, validNum, batch_size):
    K = getKValue(trainXSet, trainYSet, validXSet, validYSet, trainNum, validNum, batch_size)
    print("K:{}".format(K))
    acc = []
    errorCount = 0.0
    Num = len(testX)
    for j in range(Num):
        for i in range(len(testX[j])):
            classifierResult = knn(trainXSet, trainYSet, testX[j][i], K)
            # print("KNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
            if (classifierResult != testY[j][i]): errorCount += 1.0
        acc.append((1 - errorCount / float(len(testX[j]))) * 100)
        errorCount = 0.0
    # print("\nKNN辨识错误数量为: %d" % errorCount)
    # print("\nKNN辨识率为: %f ％" % acc)
    return acc
