from numpy import *


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    tree = dict(spInd=feat, spVal=val)
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    tree['left'] = createTree(lSet, leafType, errType, ops)
    tree['right'] = createTree(rSet, leafType, errType, ops)
    return tree


def chooseBestSplit(dataSet, leafType, errType, ops):
    tolS = ops[0]
    tolN = ops[1]
    if len((set(dataSet[:, -1].T.tolist()[0]))) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


def isTree(obj):
    return type(obj).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2


def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree, lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree, rSet)
    if not tree['left'] or not tree['right']:
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(pow(lSet[:, -1] - tree['left'], 2)) + sum(pow(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2
        errorMerge = sum(pow(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merge')
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This Matrix is singular, cannot do inverse.\n Try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(pow(Y - yHat, 2))


# forecast
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inDat=inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    trainMat = mat([[1, 2, 1.0], [2, 3, 2.0], [3, 4, 2.0]])
    myTree = createTree(trainMat, ops=(0, 1))
    print(myTree)

