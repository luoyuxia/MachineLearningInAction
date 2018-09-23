from numpy import *
from math import log


def trainNB0(train_mat, train_category):
    num_doc = len(train_mat)
    p_positive = sum(train_category) / float(num_doc)
    num_words = len(train_mat[0])
    p0Num = ones(num_words)
    p1Num = ones(num_words)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(num_doc):
        if train_category[i] == 1:
            p0Num += train_mat[i]
            p0Denom += sum(train_mat[i])
        else:
            p1Num += train_mat[i]
            p1Denom += sum(train_mat[i])
    return log(p0Num / p1Denom), log(p1Num / p1Denom), p_positive


def classify(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    return p0 if p0 > p1 else p1


