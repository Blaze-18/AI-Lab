"""
Write a report in pdf format using any Latex system after:
    ●​ building a Convolutional Neural Network (CNN) based 10 class classifier
    ●​ training and testing the classifier by using the the:
    ○​ Fashion MNIST dataset.
    ○​ MNIST English dataset.
    ○​ CIFAR-10 dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_SIZE = (28, 28)
BATCH_SIZE = 16
EPOCHS = 1

def load_mnist_data():
    """Loads the MNIST dataset and prepares it for CNN input."""
    print("\n\n====== LOADING MNIST DATASET FROM KERAS =====\n\n")
    # for cfir and fashoin mnist just change the dataset type
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # CNNs require a 4D input: (Batch, Height, Width, Channels)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
    
    return x_train, y_train, x_test, y_test

def build_cnn_model():
    """Builds a Convolutional Neural Network architecture."""
    print("\n\n====== Build CNN Model Architecture =====\n\n")
    model = keras.Sequential([
        # Standardize pixel values to [0, 1]
        layers.Rescaling(1./255, input_shape=(28, 28, 1)),
        
        # Convolutional Block 1: Detects basic edges
        layers.Conv2D(32, (3, 3), activation='relu'),  # 28,28 (3x3) = 2
        layers.MaxPooling2D((2, 2)), # Shrinks 26x26 to 13x13
        
        # Convolutional Block 2: Detects complex shapes
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), # Shrinks 11x11 to 5x5
        
        # Flattening: Prep for final decision layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')      
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    """Trains the model and evaluates performance on the test set."""
    print("\n\n====== Train and Evaluate Model =====\n\n")
    
    # Training with internal 20% validation split
    model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    ) 
    
    # Final evaluation on the unseen test set
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy

def visualize_test_results(model, x_test, y_test, num_of_img=5):
    """Visualizes model predictions on random samples from the test set."""
    indices = np.random.choice(len(x_test), num_of_img)
    images = x_test[indices]
    labels = y_test[indices]
    
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    
    plt.figure(figsize=(12, 4))
    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Actual: {labels[i]} | Pred: {pred_labels[i]}", fontsize=9)
        plt.axis("off")
    plt.savefig("cnn_mnist_result.png")
    plt.show()

def main():
    # 1. Load Data
    x_train, y_train, x_test, y_test = load_mnist_data()

    # 2. Build Model
    model = build_cnn_model()
    model.summary()

    # 3. Train and Evaluate
    loss, accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)

    # 4. Results
    print(f"\nFinal Test Accuracy: {accuracy:.4%}\nFinal Test Loss: {loss:.4f}")

    # 5. Visualize
    visualize_test_results(model, x_test, y_test, num_of_img=5)

if __name__ == "__main__":
    main()
