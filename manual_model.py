import numpy as np
import pandas as pd
import random
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):  # Check if X is a DataFrame
            X = X.values  # Convert DataFrame to numpy array
        if isinstance(y, pd.Series):  # Check if y is a Series
            y = y.values  # Convert Series to numpy array

        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:  # Tidak ada split yang valid
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # Tambahan validasi split
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        # Return None jika tak ada split yang valid
        if best_gain == 0:
            return None, None

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        if isinstance(X, pd.DataFrame):  # Check if X is a DataFrame
            X = X.values  # Convert DataFrame to numpy array
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):  # Check if X is a DataFrame
            X = X.values  # Convert DataFrame to numpy array
        if isinstance(y, pd.Series):  # Check if y is a Series
            y = y.values  # Convert Series to numpy array

        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    

import numpy as np
import random

class SVM:
    def __init__(self, max_iter=1000, C=1.0, tol=0.001):
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        self.b = 0
        self.w = None

    def fit(self, X, y):
        # konversi ke numpy
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        n_data, n_fitur = X.shape

        # inisialisasi alpha semua nol
        alpha = np.zeros(n_data)
        self.w = np.zeros(n_fitur)
        b = 0

        # loop training
        for step in range(self.max_iter):
            alpha_lama = alpha.copy()
            for i in range(n_data):
                # pilih acak j ≠ i
                j = i
                while j == i:
                    j = random.randint(0, n_data - 1)

                x_i = X[i]
                x_j = X[j]
                y_i = y[i]
                y_j = y[j]

                # hitung eta = ||xi - xj||^2
                eta = np.dot(x_i, x_i) + np.dot(x_j, x_j) - 2 * np.dot(x_i, x_j)
                if eta == 0:
                    continue

                # hitung prediksi error
                fxi = np.dot(self.w, x_i) + b
                fxj = np.dot(self.w, x_j) + b
                Ei = fxi - y_i
                Ej = fxj - y_j

                # hitung batas L dan H
                if y_i != y_j:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - self.C)
                    H = min(self.C, alpha[i] + alpha[j])

                if L == H:
                    continue

                # hitung alpha baru
                alpha_j_new = alpha[j] + y_j * (Ei - Ej) / eta

                # klip alpha_j agar tetap di dalam [L, H]
                if alpha_j_new > H:
                    alpha_j_new = H
                elif alpha_j_new < L:
                    alpha_j_new = L

                alpha_i_new = alpha[i] + y_i * y_j * (alpha[j] - alpha_j_new)

                # delta perubahan
                delta_i = alpha_i_new - alpha[i]
                delta_j = alpha_j_new - alpha[j]

                # update w
                self.w = self.w + delta_i * y_i * x_i + delta_j * y_j * x_j

                # update b (bias)
                b1 = b - Ei - y_i * delta_i * np.dot(x_i, x_i) - y_j * delta_j * np.dot(x_i, x_j)
                b2 = b - Ej - y_i * delta_i * np.dot(x_i, x_j) - y_j * delta_j * np.dot(x_j, x_j)

                if 0 < alpha_i_new < self.C:
                    b = b1
                elif 0 < alpha_j_new < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                # simpan nilai alpha baru
                alpha[i] = alpha_i_new
                alpha[j] = alpha_j_new

            # cek konvergen (norma perubahan kecil)
            perubahan = np.linalg.norm(alpha - alpha_lama)
            if perubahan < self.tol:
                break

        self.b = b

    def predict(self, X):
        X = np.array(X, dtype=float)
        hasil = np.dot(X, self.w) + self.b
        return np.sign(hasil).astype(int)