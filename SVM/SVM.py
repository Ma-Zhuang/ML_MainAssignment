from sklearn import svm


def SVMPredict(trainXSet, trainYSet, testX, testY):
    model = svm.SVC(C=10, kernel='rbf')
    model.fit(trainXSet, trainYSet)
    errorCount = 0.0
    mTest = len(testX)
    for i in range(mTest):
        classifierResult = model.predict(testX[i].reshape(1, -1))
        print("SVM得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
        if (classifierResult != testY[i]): errorCount += 1.0
    acc = (1 - errorCount / float(mTest)) * 100
    print("\n辨识错误数量为: %d" % errorCount)
    print("\n辨识率为: %f ％" % acc)
    return acc
