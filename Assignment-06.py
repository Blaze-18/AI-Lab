import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = (28, 28)
BATCH_SIZE = 16
TRAIN_DIR = 'dataset/mnist_png/train'
TEST_DIR = 'dataset/mnist_png/test'
EPOCHS = 5   # Increased from 1 for meaningful learning


# -------------------------------------------------
# LOAD TRAIN, VALIDATION AND TEST DATA
# -------------------------------------------------
def load_img_data(train_directory, test_directory):
    print("\n\n====== LOADING DATASETS ======\n\n")

    # Split TRAIN directory into train + validation
    train_ds = keras.utils.image_dataset_from_directory(
        train_directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    val_ds = keras.utils.image_dataset_from_directory(
        train_directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    # Separate TEST dataset (NO SPLITTING)
    test_ds = keras.utils.image_dataset_from_directory(
        test_directory,
        shuffle=False,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    return train_ds, val_ds, test_ds


# -------------------------------------------------
# BUILD MODEL (FCFNN)
# -------------------------------------------------
def build_model():
    print("\n\n====== BUILD MODEL ARCHITECTURE ======\n\n")

    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
def train_model(model, train_ds, val_ds):
    print("\n\n====== TRAINING MODEL ======\n\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=0
    )

    return history


# -------------------------------------------------
# EVALUATE ON TEST SET ONLY
# -------------------------------------------------
def evaluate_model(model, test_ds):
    print("\n\n====== EVALUATING ON TEST SET ======\n\n")

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    return test_loss, test_accuracy


# -------------------------------------------------
# VISUALIZE TEST PREDICTIONS
# -------------------------------------------------
def visualize_test_results(model, test_ds, num_of_img=5):

    for images, labels in test_ds.take(1):
        break

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(10, 6))

    for i in range(num_of_img):
        plt.subplot(1, num_of_img, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        plt.title(f"A:{labels[i]} | P:{pred_labels[i]}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("output/test_predictions.png")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    train_ds, val_ds, test_ds = load_img_data(TRAIN_DIR, TEST_DIR)


    model = build_model()
    model.summary()


    history = train_model(model, train_ds, val_ds)


    test_loss, test_accuracy = evaluate_model(model, test_ds)

    print(f"\nFINAL TEST ACCURACY: {test_accuracy:.4f}")
    print(f"FINAL TEST LOSS: {test_loss:.4f}")


    visualize_test_results(model, test_ds)


if __name__ == "__main__":
    main()
