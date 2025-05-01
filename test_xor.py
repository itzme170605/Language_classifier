#!/usr/bin/env python3

"""
Script to test decision tree and AdaBoost on the XOR problem
"""

import numpy as np
import pickle
import os
from lab3 import DecisionTree, AdaBoost, load_features, load_examples

# Create XOR data files
def create_xor_files():
    """Create XOR test files"""
    # Create features file
    with open('xorFeatures.txt', 'w') as f:
        f.write('aaa\nbbb\n')
    
    # Create labeled training data
    with open('xorLabel.dat', 'w') as f:
        f.write('1|aaa bbb\n')  # 1,1 -> 1
        f.write('0|aaa\n')      # 1,0 -> 0
        f.write('0|bbb\n')      # 0,1 -> 0
        f.write('1|\n')         # 0,0 -> 1
    
    # Create unlabeled test data
    with open('xorNoLabel.dat', 'w') as f:
        f.write('aaa bbb\n')    # 1,1 -> should be 1
        f.write('aaa\n')        # 1,0 -> should be 0
        f.write('bbb\n')        # 0,1 -> should be 0
        f.write('\n')           # 0,0 -> should be 1

def test_xor_directly():
    """Test decision tree and AdaBoost on XOR directly"""
    print("Testing XOR problem directly...")
    
    # Create XOR dataset
    X = np.array([
        [True, True],   # 1,1 -> 1
        [True, False],  # 1,0 -> 0
        [False, True],  # 0,1 -> 0
        [False, False]  # 0,0 -> 1
    ])
    y = np.array(['1', '0', '0', '1'])
    
    # Test decision tree
    print("\nTesting Decision Tree on XOR:")
    dt = DecisionTree(max_depth=5)
    dt.fit(X, y)
    dt_preds = dt.predict(X)
    dt_acc = np.mean(dt_preds == y)
    print(f"  Decision Tree Accuracy: {dt_acc * 100:.2f}%")
    print(f"  Predictions: {dt_preds}")
    print(f"  Expected:    {y}")
    
    # Test AdaBoost with stumps
    print("\nTesting AdaBoost with stumps on XOR:")
    ada = AdaBoost(n_estimators=10)
    ada.fit(X, y)
    ada_preds = ada.predict(X)
    ada_acc = np.mean(ada_preds == y)
    print(f"  AdaBoost Accuracy: {ada_acc * 100:.2f}%")
    print(f"  Predictions: {ada_preds}")
    print(f"  Expected:    {y}")
    
    return dt_acc, ada_acc

def test_xor_files():
    """Test using the XOR files"""
    print("\nTesting using XOR files...")
    
    # Create the XOR files
    create_xor_files()
    
    # Train and test decision tree
    print("\nTraining Decision Tree on XOR files...")
    dt = DecisionTree(max_depth=5)
    features = load_features('xorFeatures.txt')
    X, y = load_examples('xorLabel.dat', features, labeled=True)
    dt.fit(X, y)
    
    # Save the model
    with open('dt_xor.model', 'wb') as f:
        pickle.dump(dt, f)
    
    # Test the model
    X_test = load_examples('xorNoLabel.dat', features, labeled=False)
    dt_preds = dt.predict(X_test)
    expected = np.array(['1', '0', '0', '1'])
    dt_acc = np.mean(dt_preds == expected)
    print(f"  Decision Tree File Test Accuracy: {dt_acc * 100:.2f}%")
    print(f"  Predictions: {dt_preds}")
    print(f"  Expected:    {expected}")
    
    # Train and test AdaBoost
    print("\nTraining AdaBoost on XOR files...")
    ada = AdaBoost(n_estimators=10)
    ada.fit(X, y)
    
    # Save the model
    with open('ada_xor.model', 'wb') as f:
        pickle.dump(ada, f)
    
    # Test the model
    ada_preds = ada.predict(X_test)
    ada_acc = np.mean(ada_preds == expected)
    print(f"  AdaBoost File Test Accuracy: {ada_acc * 100:.2f}%")
    print(f"  Predictions: {ada_preds}")
    print(f"  Expected:    {expected}")
    
    return dt_acc, ada_acc

def run_commands():
    """Run lab3.py commands to test XOR"""
    print("\nRunning lab3.py commands for XOR test:")
    
    # Train decision tree
    dt_cmd = "python lab3.py train xorLabel.dat xorFeatures.txt dt_xor.model dt"
    print(f"\n> {dt_cmd}")
    os.system(dt_cmd)
    
    # Test decision tree
    dt_test_cmd = "python lab3.py predict xorNoLabel.dat xorFeatures.txt dt_xor.model"
    print(f"\n> {dt_test_cmd}")
    os.system(dt_test_cmd)
    
    # Train AdaBoost
    ada_cmd = "python lab3.py train xorLabel.dat xorFeatures.txt ada_xor.model ada"
    print(f"\n> {ada_cmd}")
    os.system(ada_cmd)
    
    # Test AdaBoost
    ada_test_cmd = "python lab3.py predict xorNoLabel.dat xorFeatures.txt ada_xor.model"
    print(f"\n> {ada_test_cmd}")
    os.system(ada_test_cmd)

if __name__ == "__main__":
    # Test XOR directly
    dt_acc, ada_acc = test_xor_directly()
    
    # Test using files
    file_dt_acc, file_ada_acc = test_xor_files()
    
    # Run lab3.py commands
    run_commands()
    
    # Summary
    print("\n" + "=" * 40)
    print("XOR TEST SUMMARY:")
    print("=" * 40)
    print(f"Decision Tree Direct Test:  {dt_acc * 100:.2f}%")
    print(f"Decision Tree File Test:    {file_dt_acc * 100:.2f}%")
    print(f"AdaBoost Direct Test:       {ada_acc * 100:.2f}%")
    print(f"AdaBoost File Test:         {file_ada_acc * 100:.2f}%")
    print("=" * 40)
    
    if dt_acc == 1.0 and file_dt_acc == 1.0:
        print("✓ Decision Tree can learn XOR pattern (as expected)")
    else:
        print("✗ Decision Tree FAILED to learn XOR pattern (should be 100%)")
    
    if ada_acc <= 0.5 and file_ada_acc <= 0.5:
        print("✓ AdaBoost with stumps cannot learn XOR (as expected)")
    else:
        print("✗ AdaBoost with stumps should NOT learn XOR (accuracy should be <= 50%)")