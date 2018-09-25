from numpy import *
from numpy import linalg as la


def euclidSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=False)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * num / denom


'''按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
后续计算SVD时需要将原始矩阵转换到k维空间'''


def sigmaPct(sigma, percentage):
    sigma2 = sigma ** 2  # 对sigma求平方
    sumsgm2 = sum(sigma2)  # 求所有奇异值sigma的平方和
    sumsgm3 = 0  # sumsgm3是前k个奇异值的平方和
    k = 0
    for i in sigma:
        sumsgm3 += i ** 2
        k += 1
        if sumsgm3 >= sumsgm2 * percentage:
            return k


def svdEst(dataMat, user, simMeas, item, percentage=0.9):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    k = sigmaPct(Sigma, percentage)
    Sig4 = mat(eye(k) * Sigma[:k])
    xformedItems = dataMat.T * U[:, :k] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T,
                             xformedItems[j, :].T)
        #    print('the %d and %d similarity is: %f' % (item, j, similarity))
        ratSimTotal += similarity * userRating
        simTotal += similarity
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def standEst(dataMat, user, simMeas, item, percentage=0.9):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0,
                                      dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item],
                                 dataMat[overLap, j])
            #   print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst, percentage=0.9):
    unrated_items = nonzero(dataMat[user, :].A == 0)[1]
    if len(unrated_items) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unrated_items:
        estimatedScore = estMethod(dataMat, user, simMeas, item, percentage)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda p: p[1], reverse=True)[:N]


def loadExData():
    return mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])


if __name__ == '__main__':
    testdata = loadExData()
    recommend_items = recommend(testdata, 1, N=3)
    print(recommend_items)
