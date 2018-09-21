from numpy import *
from collections import defaultdict


def classify(in_x, date_set, labels, k):
    data_size = date_set.shape[0]
    data = tile(in_x, (data_size, 1))
    data_diff = data - date_set
    distance = data_diff ** 2
    distance = distance.sum(axis=1) ** 0.5
    sort_index = distance.argsort()
    label_count_map = defaultdict(int)
    for i in range(0, k):
        label_count_map[labels[sort_index[i]]] = label_count_map[labels[sort_index[i]]] + 1
    sorted_class_count = sorted(label_count_map.items(), key=lambda item: item[1], reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    a = ((array([1, 4]) - array([3, 4])) ** 2).sum(axis=1) ** 0.5
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label_arr = ['A', 'A', 'B', 'B']
    test_x = [0, 0]
    print(classify(test_x, group, label_arr, 1))
