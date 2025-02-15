# models/cnn_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def plot_cnn_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('CNN Training Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('CNN Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    # Export the training history plot
    plt.savefig("cnn_training_history.png")
    plt.show()

def plot_cnn_metrics(y_test, y_test_pred, label_enc):
    acc   = accuracy_score(y_test, y_test_pred)
    prec  = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(8, 4))
    plt.barh(labels, values, color='salmon')
    plt.xlabel('Score')
    plt.title('CNN Evaluation Metrics on Test Set')
    plt.xlim(0, 1)
    plt.tight_layout()
    # Export the evaluation metrics plot
    plt.savefig("cnn_metrics.png")
    plt.show()

def train_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, label_enc, df_test):
    print("Training CNN model...")

    num_classes = len(np.unique(y_train))
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    input_shape = X_train.shape[1:]  # Expecting shape (features, channels)
    
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val_cat),
        verbose=1
    )
    # Plot and export the training history.
    plot_cnn_history(history)

    model.save("cnn.h5")
    print("Trained CNN model saved to cnn_genre_classifier.h5.")

    # Test Evaluation
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
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
    
    # Plot and export evaluation metrics for the test set.
    plot_cnn_metrics(y_test, y_test_pred, label_enc)

    print("CNN training complete.")
    return model
