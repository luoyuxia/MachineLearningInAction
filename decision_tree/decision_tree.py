from math import log
from collections import defaultdict


def calcShannonEnt(data_set):
    num_entries = len(data_set)
    label_count = defaultdict(int)
    for featVec in data_set:
        label_count[featVec[-1]] += 1
    shannon_ent = 0
    for label, count in label_count.items():
        prob = float(count) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def splitDataSet(data_set, axis, value):
    return_data_set = []
    for featVec in data_set:
        if featVec[axis] == value:
            reduced_vec = featVec[0:axis]
            reduced_vec.extend(featVec[axis + 1:])
            return_data_set.append(reduced_vec)
    return return_data_set


def chooseBestFeatureToSplit(data_set):
    num_data = len(data_set)
    num_features = len(data_set[0]) - 1
    base_entropy = calcShannonEnt(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_feat_values = set(feat_list)
        new_entropy = 0.0
        for v in unique_feat_values:
            split_data_set = splitDataSet(data_set, i, v)
            prob = len(split_data_set) / float(num_data)
            new_entropy += prob * calcShannonEnt(split_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def chooseMajorCnt(class_list):
    class_count = defaultdict(int)
    for c in class_list:
        class_count[c] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1],
                                reverse=True)
    return sorted_class_count[0][0]


def createTree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if len(set(class_list)) == 1:
        return class_list[0]
    if len(data_set[0]) == 1:
        return chooseMajorCnt(class_list)
    best_feature = chooseBestFeatureToSplit(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del labels[best_feature]
    feature_values = [example[best_feature] for example in data_set]
    unique_values = set(feature_values)
    for v in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][v] = createTree(splitDataSet(data_set, best_feature, v), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_tree = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key, value in second_tree.items():
        if key == test_vec[feat_index]:
            if type(value).__name__ == 'dict':
                return classify(value, feat_labels, test_vec)
            else:
                return value


def create_data_set():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'],
                [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


if __name__ == '__main__':
    data_set, labels = create_data_set()
    labels_copy = labels[:]
    mytree = createTree(data_set, labels)
    print(classify(mytree, labels_copy, [1, 1]))
