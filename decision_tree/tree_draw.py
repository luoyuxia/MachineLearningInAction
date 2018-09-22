import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args)


def getNumLeafs(my_tree):
    num_leafs = 0
    for key, value in my_tree.items():
        if type(value).__name__ == 'dict':
            num_leafs += getNumLeafs(value)
        else:
            num_leafs += 1
    return num_leafs


def getTreeDepth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    my_tree = my_tree[first_str]
    for key, value in my_tree.items():
        if type(value).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(value)
        else:
            this_depth = 1
        max_depth = max_depth if max_depth > this_depth else this_depth
    return max_depth


def retrieveTree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)


def plotTree(my_tree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(my_tree=my_tree)
    depth = getTreeDepth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(first_str, cntrPt, parentPt, decision_node)
    second_dict = my_tree[first_str]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key, value in second_dict.items():
        if type(value).__name__ == 'dict':
            plotTree(value, cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(str(value), (plotTree.xOff, plotTree.yOff), cntrPt,
                     leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(in_tree):
    flg = plt.figure(1, facecolor='white')
    flg.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(in_tree))
    plotTree.totalD = float(getTreeDepth(in_tree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(in_tree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    tree = retrieveTree(0)
    createPlot(tree)
