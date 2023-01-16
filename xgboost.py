import numpy
import pandas
from typing import List

R = 1

data = [
    [1, 0, 1, -5],
    [2, 0, 2, 5],
    [3, 1, 3, -2],
    [4, 1, 1, 2],
    [5, 1, 2, 0],
    [6, 1, 6, -5],
    [7, 1, 7, 5],
    [8, 0, 6, -2],
    [9, 0, 7, 2],
    [10, 1, 6, 0],
    [11, 1, 8, -5],
    [12, 1, 9, 5],
    [13, 0, 10, -2],
    [14, 0, 8, 2],
    [15, 1, 9, 0],
]


def sigmoid(x):
    return 1 / (1 + numpy.exp(x))


class Loss(object):

    @classmethod
    def g(cls):
        pass

    @classmethod
    def h(cls):
        pass

    @classmethod
    def loss(cls, g, h):
        return pow(g, 2) / (h + R)


class SplitInfo(object):

    def __init__(self, gain, feature_name, feature_val):
        self.gain = gain
        self.feature_name = feature_name
        self.feature_val = feature_val
        self.left_idx_list = []
        self.right_idx_list = []

    def update(self, gain, feature_name, feature_val, left_idx_list,
               right_idx_list):
        self.gain = gain
        self.feature_name = feature_name
        self.feature_val = feature_val
        self.left_idx_list = left_idx_list
        self.right_idx_list = right_idx_list


class FeaturesAndIndexes(object):

    def __init__(self, feature_name, values, indexes):
        self.feature_name = feature_name
        self.values = values
        self.indexes = indexes


class Samples(object):

    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.headers = df.columns.values
        self.features = [h for h in self.headers if h not in ["id", "y"]]
        self.labels = df["y"].to_list()
        self.n_samples = df.shape[0]
        # y_hats 初始化为0，这样 sigmoid之后的预测值就是0.5
        self.y_hats = numpy.zeros(self.n_samples, dtype=float)
        self.gradients = numpy.zeros(self.n_samples, dtype=float)
        self.hessians = numpy.zeros(self.n_samples, dtype=float)
        self.features_and_indexes: List[FeaturesAndIndexes] = []
        self._sort_by_features()
        self.update_gradients_hessians()

    def _sort_by_features(self):
        for feature_name in self.features:
            values = self.df[feature_name].to_list()
            data = [{"value": val, "index": i} for i, val in enumerate(values)]
            data = sorted(data, key=lambda x: x["value"])
            values, indexes = [], []
            for d in data:
                values.append(d["value"])
                indexes.append(d["index"])
            fi = FeaturesAndIndexes(feature_name=feature_name,
                                    values=values,
                                    indexes=indexes)
            # print(
            #     f"feature_name={feature_name}, sorted_indexes={indexes}, values={values}"
            # )
            self.features_and_indexes.append(fi)

    def update_gradients_hessians(self):
        """
        更新所有样本的一阶梯度值和二阶梯度值，每生成一棵树后需要更新一次
        p = sigmoid(y_hat)
        g = p-y
        h=p(1-p)
        """
        for i in range(len(self.labels)):
            p = sigmoid(self.y_hats[i])
            self.gradients[i] = p - self.labels[i]
            self.hessians[i] = p * (1 - p)

    def get_weight(self, idx_list):
        """计算叶子结点的权重"""
        g = sum([self.gradients[idx] for idx in idx_list])
        h = sum([self.hessians[idx] for idx in idx_list])
        weight = -g / (h + R)
        return weight

    def split_gain(self, idx_list: List) -> SplitInfo:
        """
        sorted_idx_list 还是原始数据的索引号，只不过顺序不一样了，它时按特征值排序的；
        """
        g_total = h_total = 0
        for idx in idx_list:
            g_total += self.gradients[idx]
            h_total += self.hessians[idx]
        loss_total = pow(g_total, 2) / (h_total + R)
        print(f"g_total={g_total}, h_total={h_total}, loss_total={loss_total}")
        split_info = SplitInfo(0, "", "")
        for fi in self.features_and_indexes:
            # print("calc gain for feature:", fi.feature_name)
            sorted_idx_list = [idx for idx in fi.indexes if idx in idx_list]
            g_left = h_left = 0
            g_right, h_right = g_total, h_total

            # print("split by feature: ", fi.feature_name)
            # print("sorted_idx_list=", sorted_idx_list)
            # print("sorted_values=", fi.values)

            for i, idx in enumerate(sorted_idx_list):
                if i == len(sorted_idx_list) - 1:
                    break
                g_left += self.gradients[idx]
                g_right -= self.gradients[idx]
                h_left += self.hessians[idx]
                h_right -= self.hessians[idx]
                if i + 1 < self.n_samples and fi.values[i + 1] == fi.values[i]:
                    continue
                # print(
                #     f"val={fi.values[i]}, g_left={g_left}, g_right={g_right}, h_left={h_left}, h_right={h_right}"
                # )
                loss_left = pow(g_left, 2) / (h_left + R)
                loss_right = pow(g_right, 2) / (h_right + R)
                gain = (loss_left + loss_right) - loss_total
                # print(
                #     "gain=", gain, fi.feature_name, fi.values[i],
                #     f"max_gain={split_info.gain}, {split_info.feature_name}, {split_info.feature_val}"
                # )
                if gain > split_info.gain:
                    split_info.update(gain=gain,
                                      feature_name=fi.feature_name,
                                      feature_val=fi.values[i],
                                      left_idx_list=sorted_idx_list[:i + 1],
                                      right_idx_list=sorted_idx_list[i + 1:])
        return split_info


class Node(object):

    def __init__(self, idx_list: List, depth=0):
        # 样本索引号的列表
        self.idx_list = idx_list
        # 分割点特征
        self.split_feature_name = None
        # 分割点特征值
        self.split_feature_val = None

        self.left_child = None
        self.right_child = None
        self.is_leaf = True
        # 结点权重
        self.weight = 0
        # 结点层级
        self.depth = depth

    def split_gain(self) -> bool:
        """计算结点分裂的增益，增益大于0返回True，小于0返回False"""
        pass

    def split(self, split_info: SplitInfo):
        self.is_leaf = False
        self.left_child = Node(idx_list=split_info.left_idx_list,
                               depth=self.depth + 1)
        self.right_child = Node(idx_list=split_info.right_idx_list,
                                depth=self.depth + 1)


class XGBoostTree(object):
    """
    1. 初始化root结点；
    2. 根据目标函数计算分割增益，寻找最佳分割点，并分割结点；
    3. 计算叶子结点的权重值；
    """

    def __init__(self, samples: Samples, max_depth=5):
        self.samples = samples
        self.max_depth = max_depth
        self.root = Node(idx_list=list(range(samples.n_samples)))
        self.un_split_nodes = [self.root]
        self.nodes = [self.root]

    def get_y_hats(self):
        """获取当前这棵树对每个样本的预测值"""
        y_hats = numpy.zeros(self.samples.n_samples)
        for node in self.nodes:
            if not node.is_leaf:
                continue
            for idx in node.idx_list:
                y_hats[idx] += node.weight
        # print("get y_hats=", y_hats)
        return y_hats

    def update_node_weight(self):
        """更新每个叶子结点的权重值"""
        for node in self.nodes:
            if not node.is_leaf:
                continue
            node.weight = self.samples.get_weight(idx_list=node.idx_list)

    def grow(self):
        while self.un_split_nodes:
            node = self.un_split_nodes.pop(0)
            if node.depth >= self.max_depth:
                continue
            # status = node.split_gain()
            split_info = self.samples.split_gain(idx_list=node.idx_list)
            if split_info.gain:
                print("gain=", split_info.gain)
                print(
                    "split feature=",
                    split_info.feature_name,
                    split_info.feature_val,
                )
                node.split(split_info)
                self.un_split_nodes.append(node.left_child)
                self.un_split_nodes.append(node.right_child)
                self.nodes.append(node.left_child)
                self.nodes.append(node.right_child)


if __name__ == "__main__":
    df = pandas.read_csv("vert_promoter.csv")
    # df = pandas.DataFrame(data, columns=["id", "y", "x1", "x2"])
    samples = Samples(df)
    num_trees = 5
    for i in range(num_trees):
        print(f"第{i}棵树")
        print("y_hats=", samples.y_hats)
        # print("gradients", samples.gradients)
        # print("hessians", samples.hessians)
        tree = XGBoostTree(samples=samples, max_depth=3)
        tree.grow()
        """
        树训练完成以后，还有以下步骤：
        1. 根据预测值，计算叶子结点权重；
        2. 根据叶子结点权重，计算新的预测值；
        3. 根据叶子结点权重，更新各个样本的一阶梯度和二阶梯度；
        """
        tree.update_node_weight()
        samples.y_hats += tree.get_y_hats()
        samples.update_gradients_hessians()
