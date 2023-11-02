import numpy as np

def most_common_element(iterable):
  unique_elements, counts = np.unique(iterable, return_counts=True)
  return unique_elements[np.argmax(counts)]

class Node:
  def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, *,value=None):
    self.feature = feature
    self.threshold = threshold
    self.left_node = left_node
    self.right_node = right_node
    self.value = value

  def is_leaf_node(self):
    return self.value is not None
  

class DecisionTree:
  def __init__(self, min_examples_for_split=2, max_depth=100, num_features=None):
    self.min_examples_for_split = min_examples_for_split
    self.max_depth = max_depth
    self.num_features = num_features
    self.root = None

  def fit(self, X, y):
    # If only a subset of the total number features is to be considered (useful later for implementing Random Forests),
    # then select specified number of features randomly, else consider all the features present in the dataset
    self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)
    self.root = self._grow_tree(X, y)

  def _grow_tree(self, X, y, depth=0):
    num_examples, num_features = X.shape
    num_labels = len(np.unique(y))

    # Check the criteria for stopping the growth of the tree
    if (depth >= self.max_depth or num_labels == 1 or num_examples < self.min_examples_for_split):
      return Node(value = self._most_common_label(y))

    # Choose the specified number of features randomly
    feature_indices = np.random.choice(num_features, self.num_features, replace=False)

    # Find the best feature and the best threshold value for the split on this node
    best_feature_index, best_threshold = self._best_split(X, y, feature_indices)

    # Create child left and right nodes and recursively call this function to grow the tree
    left_indices, right_indices = self._split(X[:, best_feature_index], best_threshold)
    left_node = self._grow_tree(X[left_indices, :], y[left_indices], depth+1)
    right_node = self._grow_tree(X[right_indices, :], y[right_indices], depth+1)
    return Node(best_feature_index, best_threshold, left_node, right_node)

  def _best_split(self, X, y, feature_indices):
    best_gain = -1
    best_feature_index, best_threshold = None, None
    # threshold is the value of the best feature, based on which the node is split into right and left nodes
    # it's one of the many unique values present in the best feature's column

    for feature_index in feature_indices:
      X_column = X[:, feature_index]
      thresholds = np.unique(X_column)

      for threshold in thresholds:
        # calculate the information gain
        gain = self._information_gain(y, X_column, threshold)

        # if a new maximum for the best gain is obtained then update the best gain, and the feature index and the threshold to split on.
        if gain > best_gain:
          best_gain = gain
          best_feature_index = feature_index
          best_threshold = threshold

    return best_feature_index, best_threshold

  def _split(self, X_column, split_threshold):
    left_indices = np.argwhere(X_column <= split_threshold).flatten()
    right_indices = np.argwhere(X_column > split_threshold).flatten()
    return left_indices, right_indices

  def _most_common_label(self, y):
    return most_common_element(y) # most common label present in a leaf node

  def _entropy(self, y):
    hist = np.bincount(y) # This is an array where each element at index i represents the count of occurrences
    # of the integer i in the input array. Here i is from 0 to the maximum positive integer present in the input array.

    p_s = hist / len(y) # array of all the p's, where p[i] = (hist[i] / np.sum(hist))
    return -np.sum([p * np.log(p) for p in p_s if p>0]) # p must be > 0 else log(p) is undefined

  def _information_gain(self, y, X_column, threshold):
    parent_entropy = self._entropy(y)

    # Create child nodes and calculate their weighted entropy
    left_indices, right_indices = self._split(X_column, threshold)
    if len(left_indices) == 0 or len(right_indices) == 0:
      return 0
    entropy_from_left_child, entropy_from_right_child = self._entropy(y[left_indices]), self._entropy(y[right_indices])
    child_entropy = (len(left_indices)/len(y)) * entropy_from_left_child + (len(right_indices)/len(y)) * entropy_from_right_child

    # Calculate and return the information gain for this particular split
    information_gain = parent_entropy - child_entropy
    return information_gain

  def predict(self, X):
    return np.array([self._traverse_tree(x, self.root) for x in X]) # make a label prediction by traversing the tree for every example x present in X

  def _traverse_tree(self, x, node):
    if node.is_leaf_node():
      return node.value

    # Recursively call this function to traverse the tree
    if x[node.feature] <= node.threshold:
      return self._traverse_tree(x, node.left_node)
    else:
      return self._traverse_tree(x, node.right_node)