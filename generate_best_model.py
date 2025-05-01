#!/usr/bin/env python3

"""
Script to generate best.model file using our decision tree model
"""

import pickle
import numpy as np
import sys
import os

# Import the required classes from lab3.py
from lab3 import DecisionTree, load_features, load_examples

print("Generating best.model using decision tree...")

# Load features
features = load_features('features.txt')
print(f"Loaded {len(features)} features: {', '.join(features)}")

# Load training data
try:
    X, y = load_examples('train.dat', features, labeled=True)
    print(f"Loaded {len(X)} training examples")
    
    # Check class balance
    unique_classes, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts))
    print(f"Class distribution: {class_distribution}")
    
    # Create and train the best model (decision tree with max_depth=5)
    model = DecisionTree(max_depth=5)
    model.fit(X, y)
    print("Model trained successfully")
    
    # Save the model as best.model
    with open('best.model', 'wb') as f:
        pickle.dump(model, f)
    print("best.model created successfully!")
except Exception as e:
    print(f"Error during model training: {str(e)}")
    sys.exit(1)

# Verify the model works on XOR test case
try:
    print("\nVerifying XOR test case...")
    
    # Create simple XOR dataset
    xor_features = ["aaa", "bbb"]
    xor_X = np.array([
        [True, True],   # 1,1 -> 1
        [True, False],  # 1,0 -> 0
        [False, True],  # 0,1 -> 0
        [False, False]  # 0,0 -> 1
    ])
    xor_y = np.array(['1', '0', '0', '1'])
    
    # Train a decision tree on XOR
    xor_model = DecisionTree(max_depth=5)
    xor_model.fit(xor_X, xor_y)
    
    # Test predictions
    xor_preds = xor_model.predict(xor_X)
    accuracy = np.mean(xor_preds == xor_y)
    print(f"XOR Test Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy == 1.0:
        print("XOR test passed! Decision tree can learn XOR pattern")
    else:
        print("WARNING: Decision tree failed to learn XOR pattern")
except Exception as e:
    print(f"Error during XOR verification: {str(e)}")