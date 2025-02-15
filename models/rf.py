# models/rf_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from joblib import dump

def plot_rf_metrics(y_val, y_val_pred, y_test, y_test_pred):
    # Compute metrics for validation set
    val_acc   = accuracy_score(y_val, y_val_pred)
    val_prec  = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_rec   = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1    = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    # Compute metrics for test set
    test_acc  = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_rec  = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1   = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    metrics = {
        'Val Accuracy': val_acc,
        'Test Accuracy': test_acc,
        'Val Precision': val_prec,
        'Test Precision': test_prec,
        'Val Recall': val_rec,
        'Test Recall': test_rec,
        'Val F1 Score': val_f1,
        'Test F1 Score': test_f1,
    }
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel("Score")
    plt.title("Random Forest Evaluation Metrics")
    plt.xlim(0, 1)
    plt.tight_layout()

    # Export the graph as an image file
    plt.savefig("rf_model_metrics.png")
    plt.show()


def train_rf_model(X_train, y_train, X_val, y_val, X_test, y_test, label_enc, df_test, scaler):
    print("Training RandomForestClassifier...")

    clf = RandomForestClassifier(
        n_estimators=500, max_depth=30, min_samples_split=5, 
        min_samples_leaf=2, max_features='sqrt', criterion='gini',
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    
    clf.fit(X_train, y_train)
    model_filename = "rf.joblib"
    dump(clf, model_filename)
    print(f"Trained model saved to {model_filename}.")

    # Validation Evaluation
    y_val_pred = clf.predict(X_val)
    print("Validation Classification Report:")
    unique_labels = np.unique(y_val)
    print(classification_report(y_val, y_val_pred, target_names=label_enc.inverse_transform(unique_labels), zero_division=0))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # Test Evaluation
    y_test_pred = clf.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_enc.classes_, zero_division=0))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    predicted_labels = label_enc.inverse_transform(y_test_pred)
    actual_labels = label_enc.inverse_transform(y_test)
    df_test["predicted_genre"] = predicted_labels
    df_test["actual_genre"] = actual_labels

    with open("predictions.txt", "w") as file:
        file.write("\nSample of Test Predictions vs Actual:\n")
        for idx, row in df_test.iterrows():
            file.write(f"Path: {row['path']} | Actual: {row['actual_genre']} | Predicted: {row['predicted_genre']}\n")
    
    # Plot and export evaluation metrics for comparison.
    plot_rf_metrics(y_val, y_val_pred, y_test, y_test_pred)

    print("Random Forest training complete.")
    return clf
