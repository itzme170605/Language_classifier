#!/usr/bin/env python3

"""
This script demonstrates how to test the lab3.py implementation on the XOR problem.
XOR is a classic problem that decision trees can solve, but single-level decision stumps cannot.

Usage:
1. Generate the XOR sample data (if not already provided)
2. Train with both DT and ADA algorithms
3. Test the results
"""

import subprocess
import os

# Define file paths
XOR_LABEL = "xorLabel.dat"
XOR_NO_LABEL = "xorNoLabel.dat"
XOR_FEATURES = "xorFeatures.txt"
DT_MODEL = "dt_xor.model"
ADA_MODEL = "ada_xor.model"

# Create XOR labeled data if not exists
if not os.path.exists(XOR_LABEL):
    with open(XOR_LABEL, 'w') as f:
        f.write("1|aaa bbb\n")    # 1 1 -> 1
        f.write("0|aaa\n")        # 1 0 -> 0
        f.write("0|bbb\n")        # 0 1 -> 0
        f.write("1|\n")           # 0 0 -> 1

# Create XOR unlabeled data if not exists
if not os.path.exists(XOR_NO_LABEL):
    with open(XOR_NO_LABEL, 'w') as f:
        f.write("aaa bbb\n")      # 1 1 -> expect 1
        f.write("aaa\n")          # 1 0 -> expect 0
        f.write("bbb\n")          # 0 1 -> expect 0
        f.write("\n")             # 0 0 -> expect 1

# Create XOR features if not exists
if not os.path.exists(XOR_FEATURES):
    with open(XOR_FEATURES, 'w') as f:
        f.write("aaa\n")
        f.write("bbb\n")

# Train decision tree model
print("Training decision tree model...")
subprocess.run(["python", "lab3.py", "train", XOR_LABEL, XOR_FEATURES, DT_MODEL, "dt"])

# Train AdaBoost model
print("Training AdaBoost model...")
subprocess.run(["python", "lab3.py", "train", XOR_LABEL, XOR_FEATURES, ADA_MODEL, "ada"])

# Test decision tree model
print("\nTesting decision tree model...")
dt_result = subprocess.run(["python", "lab3.py", "predict", XOR_NO_LABEL, XOR_FEATURES, DT_MODEL], 
                         capture_output=True, text=True)
print("DT Predictions:")
print(dt_result.stdout)

# Test AdaBoost model
print("\nTesting AdaBoost model...")
ada_result = subprocess.run(["python", "lab3.py", "predict", XOR_NO_LABEL, XOR_FEATURES, ADA_MODEL], 
                          capture_output=True, text=True)
print("ADA Predictions:")
print(ada_result.stdout)

print("\nExpected results:")
print("- Decision Tree should achieve 100% accuracy on XOR")
print("- AdaBoost with stumps should achieve only 50% accuracy on XOR")