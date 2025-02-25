# For numerical operations
import numpy as np
# For plotting graphs
import matplotlib.pyplot as plt

# Import evaluation metrics from sklearn
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)

# Necessary modules for building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Function to the CNN training history
def plot_cnn_history(history):
    # Create a new figure with specified dimensions
    plt.figure(figsize=(12, 4))
    
    # Create the first subplot for Loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(history.history['loss'], label='Train Loss')  # Plot training loss over epochs
    plt.plot(history.history['val_loss'], label='Val Loss')  # Plot validation loss over epochs
    plt.xlabel('Epochs')  # Label x-axis
    plt.ylabel('Loss')  # Label y-axis
    plt.title('CNN Training Loss')  # Title for the loss plot
    plt.legend()  # Add a legend to distinguish between training and validation curves
    
    # Create the second subplot for Accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(history.history['accuracy'], label='Train Acc')  # Plot training accuracy over epochs
    plt.plot(history.history['val_accuracy'], label='Val Acc')  # Plot validation accuracy over epochs
    plt.xlabel('Epochs')  # Label x-axis
    plt.ylabel('Accuracy')  # Label y-axis
    plt.title('CNN Training Accuracy')  # Title for the accuracy plot
    plt.legend()  # Add a legend for the accuracy curves
    
    plt.tight_layout()  # Adjust subplots to fit in the figure area without overlapping
    plt.savefig("cnn_training_history.png")  # Save the training history plot as an image file
    plt.show()  # Display the plot

# Functiom to plot avaluation metrics (Accuracy, Precision, Recall, F1 Score)
def plot_cnn_metrics(y_test, y_test_pred, label_enc):
    # Calculate the evaluation metrics using sklearn functions
    acc   = accuracy_score(y_test, y_test_pred)  # Compute accuracy
    prec  = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)  # Compute weighted precision
    rec   = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)  # Compute weighted recall
    f1    = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)  # Compute weighted F1 score
    
    # Create a dictionary to hold the metric names and their corresponding values
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }
    
    # Extract labels and their values from the dictionary for plotting
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # Create a new figure for the metrics plot
    plt.figure(figsize=(8, 4))
    # Create a horizontal bar plot for the metrics
    plt.barh(labels, values, color='salmon')
    plt.xlabel('Score')  # Label the x-axis
    plt.title('CNN Evaluation Metrics on Test Set')  # Title for the metrics plot
    plt.xlim(0, 1)  # Set x-axis limits between 0 and 1 (as scores are between 0 and 1)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels or titles
    plt.savefig("cnn_metrics.png")  # Save the metrics plot as an image file
    plt.show()  # Display the plot

# Function to train the CNN model and evaluate its performance on test data
def train_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, label_enc, df_test):
    print("Training CNN model...")

    # Determine the number of unique classes from the training labels
    num_classes = len(np.unique(y_train))
    
    # Convert the class labels to one-hot encoded format for training, validation, and testing
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Define the input shape for the model based on the shape of the training data (excluding the batch size)
    input_shape = X_train.shape[1:]  # For example, (timesteps, features) for 1D CNN

    # Build the CNN model using the Keras Sequential API
    model = Sequential()
    # First 1D convolutional layer: 64 filters, kernel size of 3, using ReLU activation function
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    # First max pooling layer to reduce the spatial dimensions
    model.add(MaxPooling1D(pool_size=2))
    # Second 1D convolutional layer: 128 filters, kernel size of 3, using ReLU activation function
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    # Second max pooling layer
    model.add(MaxPooling1D(pool_size=2))
    # Flatten the output of the convolutional layers to prepare it for the dense layers
    model.add(Flatten())
    # Fully connected dense layer with 128 neurons and ReLU activation function
    model.add(Dense(128, activation='relu'))
    # Dropout layer to reduce overfitting by randomly setting 50% of inputs to 0 during training
    model.add(Dropout(0.5))
    # Output layer: number of neurons equals number of classes, using softmax activation for classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with the Adam optimizer and categorical crossentropy loss function
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model using the training data, while validating on the validation data
    history = model.fit(
        X_train, y_train_cat,  # Training data and corresponding one-hot labels
        epochs=50,             # Number of epochs to train the model
        batch_size=32,         # Number of samples per gradient update
        validation_data=(X_val, y_val_cat),  # Data for validation to monitor overfitting
        verbose=1              # Verbosity mode: 1 = progress bar logging
    )
    
    # Plot and save the training history (loss and accuracy curves)
    plot_cnn_history(history)

    # Save the trained CNN model to a file for later use
    model.save("cnn.h5")
    print("Trained CNN model saved to cnn_genre_classifier.h5.")

    # Evaluate the model on the test set by first predicting class probabilities
    y_test_pred_prob = model.predict(X_test)
    # Convert the predicted probabilities to class labels by selecting the index with the highest probability
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    
    # Print a detailed classification report including precision, recall, and F1 score for each class
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_enc.classes_, zero_division=0))
    
    # Print the confusion matrix to see the correct and incorrect classifications
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Convert numerical predictions and actual labels back to their original label names
    predicted_labels = label_enc.inverse_transform(y_test_pred)
    actual_labels = label_enc.inverse_transform(y_test)
    
    # Add the predicted and actual genres as new columns in the test DataFrame for further analysis
    df_test["predicted_genre"] = predicted_labels
    df_test["actual_genre"] = actual_labels

    # Write a sample of the test predictions along with the actual labels to a text file for record keeping
    with open("predictions.txt", "w") as file:
        file.write("\nSample of Test Predictions vs Actual:\n")
        for idx, row in df_test.iterrows():
            file.write(f"Path: {row['path']} | Actual: {row['actual_genre']} | Predicted: {row['predicted_genre']}\n")
    
    # Plot and save the evaluation metrics (accuracy, precision, recall, F1 score) for the test set
    plot_cnn_metrics(y_test, y_test_pred, label_enc)

    print("CNN training complete.")
    # Return the trained model for further use or evaluation
    return model