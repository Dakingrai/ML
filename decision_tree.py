import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# [Include the DecisionTreeClassifier class code here - Node and DecisionTreeClassifier classes]

class Node:
    """Represents a node in the decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    """Decision Tree Classifier implemented from scratch"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X, y):
        """Build the decision tree"""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth if self.max_depth else False) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # If no valid split found, create leaf node
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold
        
        # Check minimum samples per leaf
        if np.sum(left_idxs) < self.min_samples_leaf or \
           np.sum(right_idxs) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Recursively build left and right subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left, right=right)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self._information_gain(y, feature_values, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, feature_values, threshold):
        """Calculate information gain for a split"""
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split the data
        left_idxs = feature_values <= threshold
        right_idxs = feature_values > threshold
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Calculate weighted average of child entropies
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        e_left = self._entropy(y[left_idxs])
        e_right = self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Information gain
        ig = parent_entropy - child_entropy
        return ig
    
    def _entropy(self, y):
        """Calculate entropy of a label distribution"""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _most_common_label(self, y):
        """Return the most common label in y"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction for a single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


# Load the breast cancer dataset
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Classes: {data.target_names}")
print(f"Class distribution: {np.bincount(y)}")
print()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print()

# Train the decision tree with different hyperparameters
print("=" * 60)
print("Training Decision Tree Classifier...")
print("=" * 60)

# Model 1: No max depth (full tree)
print("\n1. Full tree (no max_depth constraint):")
clf1 = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1)
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"   Test Accuracy: {accuracy1:.4f}")

# Model 2: Limited depth
print("\n2. Limited depth (max_depth=5):")
clf2 = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"   Test Accuracy: {accuracy2:.4f}")

# Model 3: More conservative pruning
print("\n3. Conservative pruning (max_depth=10, min_samples_split=10):")
clf3 = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5)
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print(f"   Test Accuracy: {accuracy3:.4f}")

# Detailed evaluation of the best model
print("\n" + "=" * 60)
print("Detailed Evaluation (Model 2)")
print("=" * 60)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred2))
print("\nClassification Report:")
print(classification_report(y_test, y_pred2, target_names=data.target_names))

# Compare train vs test accuracy to check for overfitting
y_train_pred2 = clf2.predict(X_train)
train_accuracy2 = accuracy_score(y_train, y_train_pred2)
print(f"\nTrain Accuracy: {train_accuracy2:.4f}")
print(f"Test Accuracy: {accuracy2:.4f}")
print(f"Difference: {train_accuracy2 - accuracy2:.4f}")

if train_accuracy2 - accuracy2 > 0.1:
    print("⚠️  Model may be overfitting (train accuracy >> test accuracy)")
else:
    print("✓ Model generalization looks good")