import numpy as np
import random


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity  # 叶子节点个数
        self.data_pointer = 0
        self.n_entries = 0
        self.tree = np.zeros(2 * capacity - 1)  # 树中总的节点个数
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):
        """Update the sampling weight
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        """Adding new data to the sumTree
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        # print ("tree_idx=", tree_idx)
        # print ("nonzero = ", np.count_nonzero(self.tree))
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        """Sampling the data
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])


class ReplayTree:
    """ReplayTree for the per(Prioritized Experience Replay) DQN. 
    """

    def __init__(self, capacity):
        self.capacity = capacity  # the capacity for memory replay
        self.tree = SumTree(capacity)
        self.abs_err_upper = 1.

        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.01
        self.abs_err_upper = 1.

    def __len__(self):
        """ return the num of storage
        """
        return self.tree.total()

    def push(self, error, sample):
        """Push the sample into the replay according to the importance sampling weight
        """
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size):
        """This is for sampling a batch data and the original code is from:
        https://github.com/rlcode/per/blob/master/prioritized_memory.py
        """
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights

    def batch_update(self, tree_idx, abs_errors):
        """Update the importance sampling weight
        """
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
