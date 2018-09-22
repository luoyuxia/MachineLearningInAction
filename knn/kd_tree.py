from numpy import *
import sys


class KdTreeNode:
    def __init__(self, data, k, level, left, right, parent):
        self.data = data
        self.dimension = k
        self.level = level
        self.left = left
        self.right = right
        self.parent = parent


class KdTree:
    def __init__(self, data_set):
        self.data_set = data_set
        self.dimension = self.data_set.shape[1]
        self.k = 0
        self.root = None

    @staticmethod
    def get_median(data):
        sort_data = sorted(data)
        median_value = sort_data[int(len(sort_data) / 2)]
        median_index = data.index(median_value)
        return median_index

    def init_kd_tree(self):
        self.root = self.get_kd_tree_node(None, self.data_set)

    @staticmethod
    def get_median_left_right_data(data, median_index, dimension):
        left_data = []
        right_data = []
        median_value = data[median_index][dimension]
        for i, d in enumerate(data):
            if i == median_index:
                continue
            if d[dimension] <= median_value:
                left_data.append(d)
            else:
                right_data.append(d)
        return left_data, right_data

    def get_kd_tree_node(self, parent, data):
        if len(data) == 0:
            return None
        k = self.k % self.dimension
        dimension_data = [row[k] for row in data]
        median_index = self.get_median(dimension_data)
        tree_node_data = data[median_index]
        kd_tree_node = KdTreeNode(tree_node_data, self.k % self.dimension, self.k, None, None, parent)
        left_data, right_data = self.get_median_left_right_data(data, median_index, k)
        self.k += 1
        left_tree_node = self.get_kd_tree_node(kd_tree_node, left_data)
        right_tree_node = self.get_kd_tree_node(kd_tree_node, right_data)
        kd_tree_node.left = left_tree_node
        kd_tree_node.right = right_tree_node
        return kd_tree_node

    def knn_nearest(self, x):
        return KdTree.nearest(self.root, x, None)

    def knn(self, x, k=1):
        kd_distances = [sys.maxsize for _ in range(0, k)]
        kd_tree_node = [None for _ in range(0, k)]
        KdTree.nearest_k(self.root, x, None, kd_distances, kd_tree_node)
        kd_tree_node = [kd_tree_node[index] for index in list(argsort(kd_distances))]
        return sorted(kd_distances), kd_tree_node

    @staticmethod
    def nearest_k(node, x, back_node, k_distance, k_tree_node):
        if node is None:
            return k_distance, k_tree_node
        leaf_node = KdTree.find_kd_leaf_node(node, x)
        distance = KdTree.cal_point_distance(x, leaf_node)
        KdTree.replace_with_tree_node(k_distance, k_tree_node, leaf_node, distance)
        node = leaf_node
        while node.parent is not None:
            children_node = node
            node = node.parent
            dimension = node.dimension
            KdTree.replace_with_tree_node(k_distance, k_tree_node, node, KdTree.cal_point_distance(x, node))
            if node is back_node:
                continue
            if max(k_distance) > node.data[dimension] - x[dimension]:
                another_node = node.right if node.left is children_node else node.left
                if another_node is not None:
                    KdTree.nearest_k(another_node, x, node, k_distance, k_tree_node)

    @staticmethod
    def replace_with_tree_node(k_distance, k_tree_node, tree_node, distance):
        max_distance_index = argmax(k_distance)
        if k_distance[max_distance_index] > distance and tree_node not in k_tree_node:
            k_distance[max_distance_index] = distance
            k_tree_node[max_distance_index] = tree_node

    @staticmethod
    def nearest(node, x, back_node):
        if node is None:
            return sys.maxsize, None
        leaf_node = KdTree.find_kd_leaf_node(node, x)
        nearest_node = leaf_node
        min_distance = KdTree.cal_point_distance(x, leaf_node)
        node = leaf_node
        while node.parent is not None:
            children_node = node
            node = node.parent
            dimension = node.dimension
            if KdTree.cal_point_distance(x, node) < min_distance:
                min_distance = KdTree.cal_point_distance(x, node)
                nearest_node = node
            if node is back_node:
                continue
            if min_distance > node.data[dimension] - x[dimension]:
                another_node = node.right if node.left is children_node else node.left
                if another_node is not None:
                    distance, nearest_node_1 = KdTree.nearest(another_node, x, node)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_node = nearest_node_1
        return min_distance, nearest_node

    @staticmethod
    def cal_point_distance(x, tree_node):
        another_point = tree_node.data
        distance = ((array(x) - array(another_point)) ** 2).sum(axis=0) ** 0.5
        return distance

    @staticmethod
    def find_kd_leaf_node(node, x):
        dimension = node.dimension
        node_value = node.data[dimension]
        if x[dimension] <= node_value:
            if node.left is not None:
                return KdTree.find_kd_leaf_node(node.left, x)
            elif node.right is not None:
                return node.right
        else:
            if node.right is not None:
                return KdTree.find_kd_leaf_node(node.right, x)
            elif node.left is not None:
                return node.left
        return node


if __name__ == '__main__':
    data_set = array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    kd_tree = KdTree(data_set)
    kd_tree.init_kd_tree()
    kd_distance, kd_node = kd_tree.knn_nearest([7, 1])
    kd_distances, kd_tree_nodes = kd_tree.knn([7, 1], 4)
