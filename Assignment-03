"""
    building FCFNNs for solving the following equations:
        i.​ y = 5x + 10
        ii.​ y = 3x2 + 5x + 10
        iii.​ y = 4x3 + 3x2 + 5x + 10
    ●​ preparing a training set, a validation set and a test set for the above equations.
    ●​ training and testing FCFNNs using your prepared data.
    ●​ plotting original y and ‘predicted y’.
    ●​ explaining the effect of "power of an independent variable" on the architecture of
      your FCFNN and the amount of training data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def equation_builder(eq_type,n):
    x = np.random.uniform(low=-10,high=10,size=n).reshape(-1,1)
    if eq_type == 1:
        y = 5*x + 10
    elif eq_type == 2:
        y = 3*x**2 + 5*x + 10
    else:
        y = 4*x**3 + 3*x**2 + 5*x + 10
        
    return x,y
    
# x = ? y = ? 

def prepare_data(x,y):
    total_n = len(x) 
    indices = np.random.permutation(total_n) 
    
    x,y = x[indices],y[indices] 

    train_percent = int(total_n * 0.7) 
    val_percent = int(total_n * 0.1)
    test_percent = total_n - train_percent - val_percent
    
    x_train = x[ : train_percent]
    y_train = y[ : train_percent]
    
    x_val = x[train_percent: train_percent+val_percent] 
    y_val = y[train_percent: train_percent+val_percent]
    
    x_test = x[train_percent + val_percent: ]
    y_test = y[train_percent + val_percent: ]
    
    return (x_train,y_train), (x_val,y_val), (x_test,y_test)

def build_model(x_train):
    normalize_layer = layers.Normalization(axis=-1)
    normalize_layer.adapt(x_train)
    
    model = keras.Sequential([
        normalize_layer,
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32,activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    return model

def train_and_test_model(model, x_train, y_train, x_val, y_val, x_test):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        batch_size=16,
        epochs=20,
        verbose=0 
    )
    
    
    predictions = model.predict(x_test) # predicted y value

    return predictions

def visualize_results(x_test, predictions, y_test):
    
    # x_test, y_test (Actual data)
    # x_test, predicted_y

    plt.figure(figsize=(10,6))
    plt.scatter(x_test,y_test,color="blue", label="Actual")
    plt.scatter(x_test, predictions, color='red', label="Predicted")
    plt.title("Results (Actual vs Predicted)")
    plt.legend()
    plt.savefig("Eq1_Results.png")
    
    print("Result updated")
    

def main():
    x,y = equation_builder(3,10000)
    
    (x_train,y_train), (x_val,y_val), (x_test,y_test) = prepare_data(x,y)
    
    model = build_model(x_train)
    
    model.summary()
    
    predictions = train_and_test_model(model,x_train,y_train,x_val,y_val,x_test)
    visualize_results(x_test, predictions, y_test)
    

if __name__ == "__main__":
    main()



    
    
