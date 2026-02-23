import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


"""
    Question: normalize the x_train and y_train values to [-1, 1]
              normalize the x_train and y_train values to [0, 1]
              
    kivabe korbo:
    
    Shutro:
    x_norm = (x - x_min) / (x_max - x_min) 
    offset = -min_val/(max_val - min_val)
    
    x_max = 10
    x_min = -10
""""


def build_model(x_train):
    # General Normalization
    normalize_layer = layers.Normalization(axis=-1)
    normalize_layer.adapt(x_train)
    
    # Jodi bole dei range --- [0 to 1]
    min_val = np.min(x_train)
    max_val = np.max(x_train)

    scale_layer = layers.Rescaling(
        scale=1./(max_val - min_val),
        offset= -min_val/(max_val - min_val)
    )
    
    # jodi bole dei range----(-1 to 1)
    min_val = np.min(x_train)
    max_val = np.max(x_train)
    scale_layer = layers.Rescaling(
        scale = 2.0 / (max_val - min_val),
        offset = - (2.0 * min_val) / (max_val - min_val) - 1.0
    )
        
    
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        #normalize_layer, used for general normalization
        #scale_layer, used if question states normalize between 0 to 1 or -1 to 1
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']  # Added evaluation metric
    )
    
    return model
