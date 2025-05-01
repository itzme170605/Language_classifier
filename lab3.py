#!/usr/bin/env python3
import sys
import numpy as np
import pickle
import math

# Constants for learning algorithms
MAX_DEPTH = 5  # Maximum depth for decision tree (must be >1 for XOR)
NUM_STUMPS = 10  # Number of stumps for AdaBoost

class Node:
    """Node class for decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index
        self.threshold = threshold  # Feature threshold
        self.left = left  # Left subtree (True branch)
        self.right = right  # Right subtree (False branch)
        self.value = value  # Label for leaf nodes

class DecisionTree:
    """Decision Tree classifier using information gain"""
    def __init__(self, max_depth=MAX_DEPTH):
        self.max_depth = max_depth
        self.root = None
        self.classes = ['nl', 'en']  # Default classes
        
    def fit(self, X, y, sample_weights=None):
        """
        Train the decision tree
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples)
            sample_weights: Weights for each sample (n_samples)
        """
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        if sample_weights is None:
            sample_weights = np.ones(len(y)) / len(y)
            
        self.classes = sorted(list(set(y)))
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        # Start recursive tree building
        self.root = self._grow_tree(X, y, sample_weights, depth=0)
        
    def predict(self, X):
        """Predict class for X"""
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        return np.array([self._predict(x, self.root) for x in X])
    
    def _predict(self, x, node):
        """Predict class for a single sample x using the decision tree"""
        if node is None:
            return self.classes[0]
            
        if node.value is not None:
            return node.value
        
        if node.feature < len(x) and x[node.feature]:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
    
    def _grow_tree(self, X, y, sample_weights, depth):
        """
        Recursively grow the decision tree
        
        Args:
            X: Features
            y: Labels
            sample_weights: Sample weights
            depth: Current depth of the tree
        
        Returns:
            A decision tree node
        """
        n_samples, n_features = X.shape
        
        # Count samples for each class
        class_counts = {}
        for i, label in enumerate(y):
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += sample_weights[i]
        
        # Stopping criteria
        if depth >= self.max_depth or len(class_counts) <= 1:
            # Create a leaf node with the majority class
            if class_counts:
                majority_class = max(class_counts, key=class_counts.get)
            else:
                majority_class = self.classes[0]
            return Node(value=majority_class)
        
        # Find the best feature to split on
        best_feature, best_gain = self._best_split(X, y, sample_weights)
        
        # If no good split, make a leaf node
        if best_feature is None:
            if class_counts:
                majority_class = max(class_counts, key=class_counts.get)
            else:
                majority_class = self.classes[0]
            return Node(value=majority_class)
        
        # Split the data
        left_indices = np.where(X[:, best_feature] == True)[0]
        right_indices = np.where(X[:, best_feature] == False)[0]
        
        # Handle empty splits (shouldn't happen with proper feature selection)
        if len(left_indices) == 0 or len(right_indices) == 0:
            if class_counts:
                majority_class = max(class_counts, key=class_counts.get)
            else:
                majority_class = self.classes[0]
            return Node(value=majority_class)
        
        # Recursive call to build subtrees
        left = self._grow_tree(
            X[left_indices], 
            y[left_indices], 
            sample_weights[left_indices], 
            depth + 1
        )
        right = self._grow_tree(
            X[right_indices], 
            y[right_indices], 
            sample_weights[right_indices], 
            depth + 1
        )
        
        return Node(feature=best_feature, threshold=True, left=left, right=right)
    
    def _best_split(self, X, y, sample_weights):
        """Find the best feature to split on"""
        best_gain = -1
        best_feature = None
        
        current_entropy = self._weighted_entropy(y, sample_weights)
        
        # Try each feature
        for feature in range(self.n_features):
            # Get indices for each branch
            left_indices = np.where(X[:, feature] == True)[0]
            right_indices = np.where(X[:, feature] == False)[0]
            
            # Skip if one side is empty
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            
            # Calculate entropy for each branch
            left_entropy = self._weighted_entropy(y[left_indices], sample_weights[left_indices])
            right_entropy = self._weighted_entropy(y[right_indices], sample_weights[right_indices])
            
            # Calculate weighted sum of entropies
            left_weight = np.sum(sample_weights[left_indices]) / np.sum(sample_weights)
            right_weight = np.sum(sample_weights[right_indices]) / np.sum(sample_weights)
            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
            
            # Calculate information gain
            gain = current_entropy - weighted_entropy
            
            # Update best gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature, best_gain
    
    def _weighted_entropy(self, y, sample_weights):
        """Calculate weighted entropy of a node"""
        # Get weighted class counts
        weighted_counts = {}
        total_weight = np.sum(sample_weights)
        
        # Handle edge case
        if total_weight == 0:
            return 0
        
        for i, label in enumerate(y):
            if label not in weighted_counts:
                weighted_counts[label] = 0
            weighted_counts[label] += sample_weights[i]
        
        # Calculate entropy
        entropy = 0
        for label in weighted_counts:
            prob = weighted_counts[label] / total_weight
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy

class AdaBoost:
    """AdaBoost ensemble classifier using decision stumps"""
    
    def __init__(self, n_estimators=NUM_STUMPS):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
        self.classes = ['nl', 'en']  # Default classes
        
    def fit(self, X, y):
        """Train the AdaBoost classifier"""
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_samples = X.shape[0]
        
        # Save classes
        self.classes = sorted(list(set(y)))
        
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        # Map classes to -1 and 1 for AdaBoost
        y_binary = np.array([1 if label == self.classes[1] else -1 for label in y])
        
        # Train weak learners
        for _ in range(self.n_estimators):
            # Train a decision stump (depth=1 tree)
            stump = DecisionTree(max_depth=1)
            stump.fit(X, y, sample_weights=weights)
            
            # Make predictions
            predictions = stump.predict(X)
            
            # Convert to binary predictions
            pred_binary = np.array([1 if p == self.classes[1] else -1 for p in predictions])
            
            # Calculate weighted error
            incorrect = (pred_binary != y_binary)
            error = np.sum(weights * incorrect) / np.sum(weights)
            
            # If error is 0.5 or worse, skip this stump
            if error >= 0.5 or error <= 0:
                # For XOR problems, it's expected that stumps can't do better than random
                if len(self.estimators) == 0:  # Make sure we have at least one stump
                    self.estimators.append(stump)
                    self.alphas.append(0.0)  # Zero weight since it's no better than random
                break
            
            # Calculate stump weight (alpha)
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            weights = weights * np.exp(-alpha * y_binary * pred_binary)
            weights = weights / np.sum(weights)  # Normalize
            
            # Add stump to the ensemble
            self.estimators.append(stump)
            self.alphas.append(alpha)
        
        # Make sure we have at least one estimator
        if len(self.estimators) == 0:
            stump = DecisionTree(max_depth=1)
            stump.fit(X, y)
            self.estimators.append(stump)
            self.alphas.append(1.0)
            
    def predict(self, X):
        """Predict class for X"""
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        if not self.estimators:
            # If no estimators, return the first class
            default_class = self.classes[0]
            return np.array([default_class] * len(X))
        
        # Sum the weighted predictions
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for alpha, estimator in zip(self.alphas, self.estimators):
            # Get predictions from this estimator
            preds = estimator.predict(X)
            
            # Convert to binary values
            pred_binary = np.array([1 if p == self.classes[1] else -1 for p in preds])
            
            # Add weighted predictions to scores
            scores += alpha * pred_binary
        
        # Convert scores back to class labels
        final_predictions = np.array([self.classes[1] if score >= 0 else self.classes[0] for score in scores])
        
        return final_predictions

def load_features(features_file):
    """Load features from file"""
    with open(features_file, 'r') as f:
        features = [line.strip() for line in f]
    return features

def load_examples(examples_file, features=None, labeled=True):
    """
    Load examples from file
    
    Args:
        examples_file: Path to examples file
        features: List of features to extract
        labeled: Whether examples are labeled or not
    
    Returns:
        X: Features
        y: Labels (if labeled=True)
    """
    X_data = []
    y_data = []
    
    with open(examples_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if labeled:
            # Handle labeled examples
            if '|' in line:
                # Standard format: "label|text"
                parts = line.split('|', 1)
                label = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else ""
                y_data.append(label)
            else:
                # Skip malformed lines
                continue
        else:
            # Unlabeled examples - just text
            text = line
        
        # Extract features if provided
        if features is not None:
            feature_vector = [feature in text for feature in features]
            X_data.append(feature_vector)
        else:
            X_data.append(text)
    
    # Convert to numpy arrays
    X = np.array(X_data)
    if labeled:
        y = np.array(y_data)
        return X, y
    else:
        return X

def train(examples_file, features_file, hypothesis_out, learning_type):
    """Train a classifier and save the model"""
    # Load features
    features = load_features(features_file)
    
    # Load examples
    X, y = load_examples(examples_file, features, labeled=True)
    
    # Create and train model
    if learning_type == "dt":
        model = DecisionTree(max_depth=MAX_DEPTH)
        model.fit(X, y)
    elif learning_type == "ada":
        model = AdaBoost(n_estimators=NUM_STUMPS)
        model.fit(X, y)
    else:
        raise ValueError(f"Unknown learning type: {learning_type}")
    
    # Save model
    with open(hypothesis_out, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {hypothesis_out}")

def predict(examples_file, features_file, hypothesis):
    """
    Load a model and predict labels for examples
    
    Args:
        examples_file: Path to examples file
        features_file: Path to features file
        hypothesis: Path to saved model
    """
    # Load features
    features = load_features(features_file)
    
    # Load examples (unlabeled)
    X = load_examples(examples_file, features, labeled=False)
    
    # Handle empty input
    if len(X) == 0:
        print("nl")  # Default prediction for empty input
        return
    
    # Load model
    with open(hypothesis, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    predictions = model.predict(X)
    
    # Print predictions (one per line)
    for label in predictions:
        print(label)

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  train <examples> <features> <hypothesisOut> <learning-type>")
        print("  predict <examples> <features> <hypothesis>")
        return
    
    command = sys.argv[1]
    
    if command == "train" and len(sys.argv) == 6:
        examples = sys.argv[2]
        features = sys.argv[3]
        hypothesis_out = sys.argv[4]
        learning_type = sys.argv[5]
        train(examples, features, hypothesis_out, learning_type)
    elif command == "predict" and len(sys.argv) == 5:
        examples = sys.argv[2]
        features = sys.argv[3]
        hypothesis = sys.argv[4]
        predict(examples, features, hypothesis)
    else:
        print("Invalid command or wrong number of arguments")
        print("Usage:")
        print("  train <examples> <features> <hypothesisOut> <learning-type>")
        print("  predict <examples> <features> <hypothesis>")

if __name__ == "__main__":
    main()