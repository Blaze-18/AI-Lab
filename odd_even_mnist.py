import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DIR = "/home/shahriar-anan/AI/tensorflow-projects/assignments/dataset/mnist_png/train"
TEST_DIR = "/home/shahriar-anan/AI/tensorflow-projects/assignments/dataset/mnist_png/test"
import tensorflow as tf
from tensorflow import keras

def load_img():
    # First load without batching to filter individual samples
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset='training',
        seed=123,
        color_mode='grayscale',
        image_size=(28, 28),
        batch_size=None  # No batching initially for clean filtering
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset='validation',
        seed=123,
        color_mode='grayscale',
        image_size=(28, 28),
        batch_size=None  # No batching initially
    )

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,
        color_mode='grayscale',
        image_size=(28, 28),
        batch_size=None  # No batching initially
    )

    # Filter individual samples (not batches)
    train_ds = train_ds.filter(lambda img, label: label % 2 == 1)
    val_ds = val_ds.filter(lambda img, label: label % 2 == 1)
    test_ds = test_ds.filter(lambda img, label: label % 2 == 1)

    """
    
    # Filter only EVEN labels (label % 2 == 0) (Remove the comments before using)
    train_ds = train_ds.filter(lambda img, label: label % 2 == 0)
    val_ds = val_ds.filter(lambda img, label: label % 2 == 0)
    test_ds = test_ds.filter(lambda img, label: label % 2 == 0)
    """

    # Now batch them
    train_ds = train_ds.batch(16)
    val_ds = val_ds.batch(16)
    test_ds = test_ds.batch(16)

    return train_ds, val_ds, test_ds

def build_model():
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),
        
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
    
def train_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=0
    )
    
    return history
    
def evaluate_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    return loss, accuracy
   
def visualize_pred(model, test_ds):

    
    for images, labels in test_ds.take(1):
        break
    

    predicted_img = model.predict(images, verbose=0)
    pred_label = np.argmax(predicted_img, axis=1)
    
    plt.figure(figsize=(15,5))
    for i in range(5):
        plt.subplot(1,5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.title(f"A: {labels[i]} | P: {pred_label[i]}")
        plt.savefig("nirjhor_img.png")

def plot_loss_acc_curve(history):
    # train
    loss = history.history["loss"]
    accuracy = history.history['accuracy']
    
    # validation
    val_loss = history.history["val_loss"]
    val_accuracy = history.history["val_accuracy"]
    
    e_range = range(1, len(loss)+1)
    
    plt.figure(figsize=(12, 6))
    # training loss vs accuracy
    plt.subplot(1,2,1)
    plt.plot(e_range, loss, color='blue', label='Train loss')
    plt.plot(e_range, accuracy, color='red', label='Train accuracy')
    plt.title("training loss vs accuracy")
    
    # validation loss vs accuracy
    plt.subplot(1,2,2)
    plt.plot(e_range, loss, label='Train loss')
    plt.plot(e_range, accuracy, label='Train accuracy')
    plt.title("validation loss vs accuracy")
    
    plt.savefig("nirjhor_curve.png")
    
    
    
def main():
    
    train_ds, val_ds, test_ds = load_img()
    
    model = build_model() 
    model.summary()
    
    history = train_model(model, train_ds, val_ds)
    
    loss, accuracy = evaluate_model(model, test_ds)
    
    print(f"Test\nLoss: {loss}")
    print(f"Accuracy: {accuracy}")
    
    visualize_pred(model, test_ds)
    plot_loss_acc_curve(history)
    
    print("Image saved")  
    
    
main()
