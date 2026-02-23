import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BATCH_SIZE = 16
EPOCHS = 3


# -------------------------------------------------
# LOAD DATASET (MNIST ONLY)
# -------------------------------------------------
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
    input_shape = (28, 28, 1)

    return x_train, y_train, x_test, y_test, input_shape


# -------------------------------------------------
# BUILD CNN MODEL (Dynamic Input Shape)
# -------------------------------------------------
def build_cnn_model(input_shape):
    print("\n\n====== Build CNN Model Architecture =====\n\n")

    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Fully Connected Section
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -------------------------------------------------
# TRAIN & EVALUATE
# -------------------------------------------------
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    print("\n\n====== Train and Evaluate Model =====\n\n")

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return history, loss, accuracy

def plot_training_curves(history):
    print("\n\n====== Plotting Accuracy & Loss Curves =====\n\n")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # -------- Accuracy Plot --------
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc)
    plt.plot(epochs_range, val_acc)
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])

    # -------- Loss Plot --------
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss)
    plt.plot(epochs_range, val_loss)
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])

    plt.tight_layout()
    plt.savefig("output/training_curves.png")
    
# -------------------------------------------------
# VISUALIZATION (Handles RGB & Grayscale)
# -------------------------------------------------
def visualize_test_results(model, x_test, y_test, num_of_img=5):
    indices = np.random.choice(len(x_test), num_of_img)
    images = x_test[indices]
    labels = y_test[indices]

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(12, 4))

    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)

        if images.shape[-1] == 1:
            plt.imshow(images[i].squeeze(), cmap="gray")
        else:
            plt.imshow(images[i].astype("uint8"))

        plt.title(f"A:{labels[i]} | P:{pred_labels[i]}", fontsize=9)
        plt.axis("off")

    plt.savefig("output/cnn_result.png")

# -------------------------------------------------
# TEST MODEL ON ONLY ODD OR EVEN DIGITS (MNIST)
# -------------------------------------------------
def predict_odd_even(model, x_test, y_test, digit_type="even", num_of_img=5):
    """
    digit_type: "even" or "odd"
    """

    print(f"\n\n====== Testing on {digit_type.upper()} digits only ======\n\n")

    if digit_type == "even":
        mask = (y_test % 2 == 0)
    elif digit_type == "odd":
        mask = (y_test % 2 != 0)
    else:
        raise ValueError("digit_type must be 'even' or 'odd'")

    # Filter dataset
    x_filtered = x_test[mask]
    y_filtered = y_test[mask]

    # Randomly choose samples
    indices = np.random.choice(len(x_filtered), num_of_img)
    images = x_filtered[indices]
    labels = y_filtered[indices]

    # Predict
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    # Plot
    plt.figure(figsize=(12, 4))

    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"A:{labels[i]} | P:{pred_labels[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"output/{digit_type}_digits_prediction.png")

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    x_train, y_train, x_test, y_test, input_shape = load_data()

    model = build_cnn_model(input_shape)
    model.summary()

    history, loss, accuracy = train_and_evaluate(
        model, x_train, y_train, x_test, y_test
    )

    print(f"\nFinal Test Accuracy: {accuracy:.4%}")
    print(f"Final Test Loss: {loss:.4f}")

    plot_training_curves(history)  
    visualize_test_results(model, x_test, y_test)
    
    predict_odd_even(model, x_test, y_test, digit_type="even")
    predict_odd_even(model, x_test, y_test, digit_type="odd")


if __name__ == "__main__":
    main()
