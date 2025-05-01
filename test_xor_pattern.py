#!/usr/bin/env python3

"""
Script to directly test the XOR pattern with the lab3.py implementation
This will create the exact XOR pattern used in the autograder tests
"""

import os
import numpy as np
import pickle
from lab3 import DecisionTree, AdaBoost, load_features

# Function to analyze the XOR pattern
def analyze_xor_pattern():
    """
    Analyze the XOR pattern in the test files to understand what it's testing
    """
    # Load features
    features = load_features('xorFeatures.txt')
    print(f"XOR Features: {features}")
    
    # Check labels in xorLabel.dat
    patterns = {}
    with open('xorLabel.dat', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
                
            label, text = line.split('|', 1)
            text = text.strip()
            
            if text not in patterns:
                patterns[text] = []
            
            if label not in patterns[text]:
                patterns[text].append(label)
    
    print("\nXOR Pattern Analysis:")
    for text, labels in patterns.items():
        feature_pattern = []
        for feature in features:
            feature_pattern.append(feature in text)
            
        print(f"Text: '{text}' -> Features: {feature_pattern} -> Labels: {', '.join(labels)}")

# Function to create a direct XOR test
def test_xor_direct():
    """
    Directly test the XOR pattern with decision tree and AdaBoost
    """
    # XOR testing features
    features = ['aaa', 'bbb']
    
    # Create the XOR training data
    # The pattern in the XOR data:
    # - 'nl' when both features match: "aaa aaa" or "bbb bbb"
    # - 'en' when features don't match: "aaa bbb" or "ccc ccc"
    X = np.array([
        [True, False],   # "aaa"
        [False, True],   # "bbb"
        [True, True],    # "aaa bbb"
        [False, False]   # ""
    ])
    y = np.array(['nl', 'nl', 'en', 'en'])
    
    # Test decision tree
    print("\nTesting Decision Tree on XOR pattern:")
    dt = DecisionTree(max_depth=5)
    dt.fit(X, y)
    dt_preds = dt.predict(X)
    dt_accuracy = np.mean(dt_preds == y) * 100
    
    print(f"  Decision Tree (depth={dt.max_depth}) Accuracy: {dt_accuracy:.2f}%")
    print("  Predictions:")
    for i, (x, pred, true) in enumerate(zip(X, dt_preds, y)):
        print(f"    Input: {x} -> Predicted: {pred}, Actual: {true}")
    
    # Test AdaBoost
    print("\nTesting AdaBoost on XOR pattern:")
    ada = AdaBoost(n_estimators=10)
    ada.fit(X, y)
    ada_preds = ada.predict(X)
    ada_accuracy = np.mean(ada_preds == y) * 100
    
    print(f"  AdaBoost Accuracy: {ada_accuracy:.2f}%")
    print("  Predictions:")
    for i, (x, pred, true) in enumerate(zip(X, ada_preds, y)):
        print(f"    Input: {x} -> Predicted: {pred}, Actual: {true}")
    
    # Save models for testing with the command-line interface
    with open('dt_xor_direct.model', 'wb') as f:
        pickle.dump(dt, f)
    with open('ada_xor_direct.model', 'wb') as f:
        pickle.dump(ada, f)
    
    print("\nModels saved as dt_xor_direct.model and ada_xor_direct.model")

if __name__ == "__main__":
    # Check if XOR files exist
    if not os.path.exists('xorFeatures.txt') or not os.path.exists('xorLabel.dat'):
        print("Creating XOR test files...")
        # Create features file
        with open('xorFeatures.txt', 'w') as f:
            f.write('aaa\nbbb\n')
        
        # Create labeled data
        with open('xorLabel.dat', 'w') as f:
            # Write many examples to match the test format
            for _ in range(512):
                f.write('nl|aaa aaa\n')
                f.write('nl|bbb bbb\n')
                f.write('en|aaa bbb\n')
                f.write('en|ccc ccc\n')
        
        # Create unlabeled data
        with open('xorNoLabel.dat', 'w') as f:
            for _ in range(512):
                f.write('aaa aaa\n')
                f.write('bbb bbb\n')
                f.write('aaa bbb\n')
                f.write('ccc ccc\n')
    
    # Analyze and test
    analyze_xor_pattern()
    test_xor_direct()
    
    print("\nNow run the following commands to test with the command-line interface:")
    print("python lab3.py train xorLabel.dat xorFeatures.txt dt_xor.model dt")
    print("python lab3.py predict xorNoLabel.dat xorFeatures.txt dt_xor.model")
    print("python lab3.py train xorLabel.dat xorFeatures.txt ada_xor.model ada")
    print("python lab3.py predict xorNoLabel.dat xorFeatures.txt ada_xor.model")