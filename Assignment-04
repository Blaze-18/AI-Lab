import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


"""
    Write a report in pdf format using any Latex system after:
    ●​ building an FCFNN based classifier according to your preferences about the
    number of hidden layers and neurons in the hidden layers.
    ●​ training and testing your FCFNN based classifier using the:
    ○​ Fashion MNIST dataset.
    ○​ MNIST English dataset.
    ○​ CIFAR-10 dataset.
"""


IMG_SIZE = (28, 28)
BATCH_SIZE = 16
EPOCHS = 1

def load_mnist_data():
    print("\n\n====== LOADING MNIST DATASET FROM KERAS =====\n\n")
    
    # 1. Load built-in data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 2. Reshape to add channel dimension (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
    
    return x_train, y_train, x_test, y_test

def build_model():
    print("\n\n====== Build Model Architecture =====\n\n")
    model = keras.Sequential([
        # Standardizes pixel values to [0, 1]
        layers.Rescaling(
            1./255, 
            input_shape=(28, 28, 1)
        ),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),      
    ])
    # 1--> 0.6 2 --> 0.9 3-->0.5 ....
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    print("\n\n====== Train Model with Data =====\n\n")
    
    # Using validation_split here instead of a separate ds object
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    ) 
    
    # Get final validation stats from history
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return loss, accuracy

def visualize_test_results(model, x_test, y_test, num_of_img=5):
    print("\n\n====== Visualizing Results =====\n\n")
    
    # Pick random indices from the test set
    indices = np.random.choice(len(x_test), num_of_img) 
    images = x_test[indices] 
    labels = y_test[indices] 
    
    # Predict (Rescaling is inside the model, so we pass raw images)
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1) 
    
    plt.figure(figsize=(12, 4))
    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)
        # Squeeze to remove the (1) channel for plotting
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Actual:{labels[i]} | Pred:{pred_labels[i]}", fontsize=9)
        plt.axis("off")

    plt.savefig("result.png")
    

def main():

    x_train, y_train, x_test, y_test = load_mnist_data()

    # 2. Build Model
    model = build_model()
    model.summary()

    # 3. Train 
    loss, accuracy = train_model(model, x_train, y_train, x_test, y_test)

    # 4. Show Accuracy
    print(f"\nFinal Validation Accuracy: {accuracy:.4f}\nFinal Validation Loss: {loss:.4f}")

    # 5. Show Predictions on Test Set
    visualize_test_results(model, x_test, y_test, num_of_img=5)

if __name__ == "__main__":
    main()
