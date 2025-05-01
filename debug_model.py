#!/usr/bin/env python3
"""
Debug script to evaluate model predictions and analyze potential biases
"""

import sys
import numpy as np
import pickle
from lab3 import load_features, load_examples

def analyze_features(train_file, features_file):
    """Analyze how distinctive each feature is for language classification"""
    features = load_features(features_file)
    X, y = load_examples(train_file, features, labeled=True)
    
    # Count occurrences of each feature by language
    en_indices = np.where(y == 'en')[0]
    nl_indices = np.where(y == 'nl')[0]
    
    en_counts = np.sum(X[en_indices], axis=0)
    nl_counts = np.sum(X[nl_indices], axis=0)
    
    print(f"Feature analysis for {len(en_indices)} English and {len(nl_indices)} Dutch examples:")
    print("{:<15} {:<10} {:<10} {:<10}".format("Feature", "English", "Dutch", "Ratio"))
    print("-" * 50)
    
    for i, feature in enumerate(features):
        en_pct = en_counts[i] / len(en_indices) * 100
        nl_pct = nl_counts[i] / len(nl_indices) * 100
        
        # Calculate ratio (how much more common in one language)
        if en_pct > nl_pct and nl_pct > 0:
            ratio = en_pct / nl_pct
            ratio_str = f"{ratio:.2f}x EN"
        elif nl_pct > en_pct and en_pct > 0:
            ratio = nl_pct / en_pct
            ratio_str = f"{ratio:.2f}x NL"
        else:
            ratio_str = "N/A"
            
        print("{:<15} {:<10.2f}% {:<10.2f}% {:<10}".format(
            feature, en_pct, nl_pct, ratio_str))

def test_model(model_file, test_file, features_file, labeled=True):
    """Test a model and analyze its predictions"""
    # Load model, features, and test data
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    features = load_features(features_file)
    
    if labeled:
        X, y_true = load_examples(test_file, features, labeled=True)
    else:
        X = load_examples(test_file, features, labeled=False)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Count predictions by class
    en_count = np.sum(y_pred == 'en')
    nl_count = np.sum(y_pred == 'nl')
    
    print(f"\nPrediction distribution:")
    print(f"English: {en_count} ({en_count / len(y_pred) * 100:.2f}%)")
    print(f"Dutch: {nl_count} ({nl_count / len(y_pred) * 100:.2f}%)")
    
    if labeled:
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Calculate accuracy by class
        en_indices = np.where(y_true == 'en')[0]
        nl_indices = np.where(y_true == 'nl')[0]
        
        en_accuracy = np.mean(y_pred[en_indices] == y_true[en_indices]) if len(en_indices) > 0 else 0
        nl_accuracy = np.mean(y_pred[nl_indices] == y_true[nl_indices]) if len(nl_indices) > 0 else 0
        
        print(f"\nOverall accuracy: {accuracy * 100:.2f}%")
        print(f"English accuracy: {en_accuracy * 100:.2f}%")
        print(f"Dutch accuracy: {nl_accuracy * 100:.2f}%")
        
        # Show confusion matrix
        print("\nConfusion Matrix:")
        print("{:<10} {:<10} {:<10}".format("", "Pred EN", "Pred NL"))
        print("-" * 30)
        print("{:<10} {:<10} {:<10}".format(
            "True EN", 
            np.sum((y_true == 'en') & (y_pred == 'en')),
            np.sum((y_true == 'en') & (y_pred == 'nl'))))
        print("{:<10} {:<10} {:<10}".format(
            "True NL", 
            np.sum((y_true == 'nl') & (y_pred == 'en')),
            np.sum((y_true == 'nl') & (y_pred == 'nl'))))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python debug_model.py analyze_features <train_file> <features_file>")
        print("  python debug_model.py test_model <model_file> <test_file> <features_file> [labeled]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "analyze_features" and len(sys.argv) == 4:
        train_file = sys.argv[2]
        features_file = sys.argv[3]
        analyze_features(train_file, features_file)
    elif command == "test_model" and len(sys.argv) >= 5:
        model_file = sys.argv[2]
        test_file = sys.argv[3]
        features_file = sys.argv[4]
        
        labeled = True
        if len(sys.argv) >= 6 and sys.argv[5].lower() == "false":
            labeled = False
        
        test_model(model_file, test_file, features_file, labeled)
    else:
        print("Invalid command or wrong number of arguments")