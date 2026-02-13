import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg16

"""
Write a report on how feature maps of different convolutional layers look when you pass
your favourite image through your three favourite pre-trained CNN classifiers..
"""

# Load and prepare image
def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


# Create model that outputs feature maps
def create_feature_model():
    model = vgg16.VGG16(weights="imagenet")

    layer_outputs = [
        model.get_layer("block1_conv1").output,
        model.get_layer("block3_conv1").output,
        model.get_layer("block5_conv1").output,
    ]

    feature_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return feature_model


# Show feature maps
def show_maps(feature_maps):
    plt.figure(figsize=(12, 4))

    for i in range(len(feature_maps)):
        plt.subplot(1, 3, i + 1)

        # show first channel only
        plt.imshow(feature_maps[i][0, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.savefig("feature_maps")


# Main function
def main():
    img = load_image("cat.jpg")
    model = create_feature_model()
    maps = model.predict(img)
    show_maps(maps)


# Run
main()
