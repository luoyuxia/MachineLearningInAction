from numpy import *
import random


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatIn * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent(dataMatrix, classLabels):
    m, n = shape(mat(dataMatrix))
    alpha = 0.001
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(weights * dataMatrix[i]))
        error = classLabels[i] - h
        weights = weights + alpha * dataMatrix[i] * error
    return weights


def stocGradAscent(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randomIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(weights * dataMatrix[randomIndex]))
            error = classLabels[randomIndex] - h
            weights = weights + alpha * dataMatrix[randomIndex] * error
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(inX * weights)
    return 1 if prob > 0.5 else 0
