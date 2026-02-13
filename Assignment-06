"""
    Write a report in pdf format using any Latex system after:
        ●​ Prepare an English handwritten digit dataset by collecting hand written data and
        splitting into the training set and test.●​ Retrain FCFNN using your training set with the training set of the MNIST English
        digit dataset.
        ●​ Evaluate your FCFNN using your test set along with the test set of the MNIST
        English dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = (28,28)
BATCH_SIZE = 16
TRAIN_DIR = 'dataset/mnist_png/train'
TEST_DIR = 'dataset/mnist_png/test'
EPOCHS = 1


def load_img_data(directory):
    print("\n\n======LOADING IMAGE=====\n\n")
    
    train_ds = keras.utils.image_dataset_from_directory(
        directory,
        validation_split = 0.2, 
        seed=123,
        subset="training",
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=16
    )
    
    return train_ds, val_ds
    
def build_model():
    
    print("\n\n======Build Model Architecture=====\n\n")
    model = keras.Sequential([
        layers.Rescaling(
            1./255,
            input_shape=(28,28,1)
        ),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),      
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
    
def train_model(model, train_ds, val_ds):
    print("\n\n======Train Model with data======\n\n")
    
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    ) 
    
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    
    return loss, accuracy

def visualize_test_results(model, num_of_img=5):
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        color_mode='grayscale',
        image_size=IMG_SIZE,
        batch_size=num_of_img,
        shuffle=True
    )
    
    for images, labels in test_ds.take(1):
        break
    
    images = images / 255.0
    
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    
    plt.figure(figsize=(10,6))
    
    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Actual:{labels[i].numpy()} | Pred:{pred_labels[i]}", fontsize=8)
        plt.axis("off")

    plt.savefig("result")

def main():
    # 1. Load Data
    train_ds, val_ds = load_img_data(TRAIN_DIR)

    # 2. Build Model
    model = build_model()
    model.summary()

    # 3. Train
    loss, accuracy = train_model(model,train_ds,val_ds)

    # 4. Show Accuracy
    print(f"Model Accuracy: {accuracy:.4f}\nLoss: {loss:.4f}")

    # 5. Show Predictions on 1–2 test images
    visualize_test_results(model, num_of_img=5)


if __name__ == "__main__":
    main()
