from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import time


def svmValid(trainXSet, trainYSet, validXSet, validYSet, trainNum, validNum, batch_size):
    errorCount = 0.0
    acc = []
    k = []
    mTest = np.random.randint(0, validNum, 500)
    for j in range(0, 50):
        k.append((j + 1) / 2500)  # trainNum // batch_size
        model = svm.SVC(C=10, kernel='rbf', gamma=(j + 1) / 2500)
        s = time.time()
        model.fit(trainXSet[:2500], trainYSet[:2500])
        print("执行了 ", j)
        print("SVM fit time:{}".format(time.time() - s))
        for i in range(len(mTest)):
            classifierResult = model.predict(validXSet[mTest[i]].reshape(1, -1))
            # print("KNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
            if (classifierResult != validYSet[mTest[i]]): errorCount += 1.0
        acc.append(((1 - errorCount / float(len(mTest))) * 100))
        errorCount = 0.0
        indexTmp = np.argwhere(acc == np.amax(acc))
        index = []
        for i in range(len(indexTmp)):
            index.append(indexTmp[i][0] + 1)
    plt.plot(k, acc)
    plt.title('SVM Correct rate', fontsize=24)
    plt.xlabel('Gama', fontsize=14)
    plt.ylabel('Correct rate(%)', fontsize=14)
    plt.show()
    print("\nValid SVM辨识率为: %f ％" % np.mean(acc))
    print(index)
    return int(np.median(index))


def fit(trainXSet, trainYSet, gama):
    print("start SVM")
    print(gama / len(trainXSet))
    model = svm.SVC(C=10, kernel='rbf', gamma=gama / len(trainXSet))
    model.fit(trainXSet, trainYSet)
    print("finished SVM")
    return model


def SVMPredict(model, testX, testY):
    errorCount = 0.0
    mTest = len(testX)
    for i in range(mTest):
        classifierResult = model.predict(testX[i].reshape(1, -1))
        # print("SVM得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
        if (classifierResult != testY[i]): errorCount += 1.0
    acc = (1 - errorCount / float(mTest)) * 100
    # print("\nSVM辨识错误数量为: %d" % errorCount)
    # print("\nSVM辨识率为: %f ％" % acc)
    return acc
