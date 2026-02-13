"""
    ​Write a report in pdf format using any Latex system after:
        ●​ training a binary classifier, based on the pre-trained VGG16, by transfer learning
        and fine tuning.
        ●​ showing the effect of fine-tuning:
        i.​ whole pre-trained VGG16
        ii.​ partial pre-trained VGG16
"""


# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# ==========================================================
# 2. LOAD DATA (CREATE VALIDATION FROM TRAIN)
# ==========================================================
def load_data():

    img_size = (224, 224)
    batch_size = 32

    # 80% training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/train",
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=123
    )

    # 20% validation data (taken from train folder)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/train",
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    # Test dataset (separate folder)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/test",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, val_ds, test_ds


# ==========================================================
# 3. CREATE MODEL WITH DIFFERENT FINE-TUNING OPTIONS
# ==========================================================
def create_model(trainable_layers=None):
    """
    trainable_layers:
        None: freeze all VGG16 layers  (transfer learning only)
        int: fine-tune last N layers  (partial fine-tuning)
        "all":  fine-tune whole VGG16    (full fine-tuning)
    """

    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))

    # Freeze everything first
    for layer in base_model.layers:
        layer.trainable = False

    # ---------- Fine-tuning options ----------
    if trainable_layers == "all":
        base_model.trainable = True

    elif isinstance(trainable_layers, int):
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    # -----------------------------------------

    # Simple classifier head
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=x)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, train_ds, val_ds, title):

    print("\n==============================")
    print("Training:", title)
    print("==============================\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1
    )

    return history


# ==========================================================
# 5. EVALUATE ON TEST DATA
# ==========================================================
def evaluate_model(model, test_ds, title):

    loss, acc = model.evaluate(test_ds, verbose=0)

    print(f"\nTest Accuracy ({title}): {acc:.4f}")
    return acc


# ==========================================================
# 6. PLOT VALIDATION ACCURACY COMPARISON
# ==========================================================
def plot_histories(histories, labels):

    plt.figure()

    for h, label in zip(histories, labels):
        plt.plot(h.history["val_accuracy"], label=label)

    plt.title("Fine-Tuning Comparison (Validation Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


# ==========================================================
# 7. MAIN FUNCTION
# ==========================================================
def main():

    # Load datasets
    train_ds, val_ds, test_ds = load_data()

    histories = []
    labels = []

    # --------------------------------------------------
    # A. Transfer Learning (freeze all VGG16 layers)
    # --------------------------------------------------
    model_frozen = create_model(trainable_layers=None)
    h1 = train_model(model_frozen, train_ds, val_ds, "Frozen VGG16")
    evaluate_model(model_frozen, test_ds, "Frozen")
    histories.append(h1)
    labels.append("Frozen")

    # --------------------------------------------------
    # B. Partial Fine-Tuning (last 4 layers)
    # --------------------------------------------------
    model_partial = create_model(trainable_layers=4)
    h2 = train_model(model_partial, train_ds, val_ds, "Partial Fine-Tuning")
    evaluate_model(model_partial, test_ds, "Partial")
    histories.append(h2)
    labels.append("Partial")

    # --------------------------------------------------
    # C. Full Fine-Tuning (entire VGG16)
    # --------------------------------------------------
    model_full = create_model(trainable_layers="all")
    h3 = train_model(model_full, train_ds, val_ds, "Full Fine-Tuning")
    evaluate_model(model_full, test_ds, "Full")
    histories.append(h3)
    labels.append("Full")

    # Plot comparison graph
    plot_histories(histories, labels)


# ==========================================================
# 8. RUN PROGRAM
# ==========================================================
if __name__ == "__main__":
    main()
