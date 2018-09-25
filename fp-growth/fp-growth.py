from numpy import *
from collections import defaultdict


class FPTreeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode
        self.nodeLink = None
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSupport=1):
    headerTable = defaultdict(int)
    for tran, count in dataSet.items():
        for item in tran:
            headerTable[item] = headerTable[item] + count
    for item in list(headerTable.keys()):
        if headerTable[item] < minSupport:
            del headerTable[item]
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for item, count in headerTable.items():
        headerTable[item] = [count, None]
    retTree = FPTreeNode('Null Set', 1, None)
    for tran, count in dataSet.items():
        localSet = {}
        for item in tran:
            if item in freqItemSet:
                localSet[item] = headerTable[item][0]
        if len(localSet) > 0:
            orderedItems = [item for item, count in sorted(localSet.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    item = items[0]
    if item in inTree.children:
        inTree.children[item].count += count
    else:
        inTree.children[item] = FPTreeNode(item, count, inTree)
        if headerTable[item][1] is None:
            headerTable[item][1] = inTree.children[item]
        else:
            updateHeader(headerTable[item][1], inTree.children[item])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[item], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpleDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    reDict = {}
    for tran in dataSet:
        reDict[frozenset(tran)] = 1
    return reDict


def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    condPats = dict()
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(headerTable, minSupport, prefix, freqItemList):
    items = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[0], reverse=True)]
    for item in items:
        prefixCopy = prefix.copy()
        prefixCopy.add(item)
        freqItemList.append(prefixCopy)
        condPats = findPrefixPath(headerTable[item][1])
        tree, header = createTree(condPats, minSupport)
        if header is not None:
            mineTree(header, minSupport, prefixCopy, freqItemList)


if __name__ == '__main__':
    '''
    rootNode = FPTreeNode('pyramid', 9, None)
    rootNode.children['eye'] = FPTreeNode('eye', 13, None)
    rootNode.children['phoenix'] = FPTreeNode('phoenix', 3, None)
    rootNode.disp()
    '''
    initSet = createInitSet(loadSimpleDat())
    tree, hearTab = createTree(initSet, 3)
    freqItems = []
    mineTree(hearTab, 3, set([]), freqItems)
    print(freqItems)
    tree.disp()
