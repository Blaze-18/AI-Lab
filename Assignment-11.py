"""
    Discuss the feature extraction power of your favorite CNN pretrained by the ImageNet
    dataset before and after transfer learning by the MNIST digit dataset after plotting high
    dimensional feature vectors on 2D plane using the following two dimension reduction
    techniques:
        ●​ Principal Component Analysis (PCA)
        ●​ t-distributed Stochastic Neighbor Embedding (t-SNE)
"""

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# 2. LOAD AND PREPARE MNIST DATA
# ==========================================================
def load_mnist(sample_size=500):
    """
    Loads MNIST and converts to RGB 224x224 for VGG16.
    Only small sample is used → faster for beginners.
    """
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    x = x_train[:sample_size]
    y = y_train[:sample_size]

    # Convert grayscale
    x = np.stack([x]*3, axis=-1)                # (N, 28, 28, 3)
    x = tf.image.resize(x, (224, 224)).numpy()  # (N, 224, 224, 3)

    return x, y


# ==========================================================
# 3. CREATE FEATURE EXTRACTOR MODEL
# ==========================================================
def create_feature_extractor(trainable=False):
    """
    Returns VGG16 model without classifier.
    trainable=False before transfer learning
    trainable=True after transfer learning
    """
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(224, 224, 3))

    base.trainable = trainable

    # Global average pooling → feature vector
    x = layers.GlobalAveragePooling2D()(base.output)

    model = models.Model(inputs=base.input, outputs=x)
    return model


# ==========================================================
# 4. SIMPLE TRANSFER LEARNING ON MNIST
# ==========================================================
def transfer_learn_model(x, y):
    """
    Fine-tunes VGG16 on MNIST digits.
    """
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(224, 224, 3))

    base.trainable = True   # full fine-tuning

    x_out = layers.GlobalAveragePooling2D()(base.output)
    x_out = layers.Dense(128, activation="relu")(x_out)
    x_out = layers.Dense(10, activation="softmax")(x_out)

    model = models.Model(base.input, x_out)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x, y, epochs=1, batch_size=32, verbose=1)  # only 1 epoch

    # Return feature extractor part
    feature_model = models.Model(model.input,
                                 model.layers[-3].output)

    return feature_model


# ==========================================================
# 5. EXTRACT HIGH-DIMENSIONAL FEATURES
# ==========================================================
def extract_features(model, x):
    """
    Returns feature vectors from CNN.
    """
    features = model.predict(x, verbose=0)
    return features


# ==========================================================
# 6. DIMENSION REDUCTION (PCA + t-SNE)
# ==========================================================
def reduce_dimensions(features):
    """
    Converts high-dimensional features → 2D.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    tsne_result = tsne.fit_transform(features)

    return pca_result, tsne_result


# ==========================================================
# 7. PLOT RESULTS
# ==========================================================
def plot_2d(points, labels, title, fig_name):
    """
    Scatter plot of 2D features.
    """
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=5)
    plt.title(title)
    plt.colorbar()
    plt.savefig(fig_name)


# ==========================================================
# 8. MAIN FUNCTION
# ==========================================================
def main():

    # Load MNIST
    x, y = load_mnist()

    # -------- BEFORE TRANSFER LEARNING --------
    model_before = create_feature_extractor(trainable=False)
    features_before = extract_features(model_before, x)
    pca_before, tsne_before = reduce_dimensions(features_before)

    # -------- AFTER TRANSFER LEARNING --------
    model_after = transfer_learn_model(x, y)
    features_after = extract_features(model_after, x)
    pca_after, tsne_after = reduce_dimensions(features_after)

    # -------- PLOTS --------
    plot_2d(pca_before, y, "PCA Before Transfer Learning", "pca_before")
    plot_2d(tsne_before, y, "t-SNE Before Transfer Learning", "tsne_before")

    plot_2d(pca_after, y, "PCA After Transfer Learning", "pca_after")
    plot_2d(tsne_after, y, "t-SNE After Transfer Learning", "tsne_after")


if __name__ == "__main__":
    main()
