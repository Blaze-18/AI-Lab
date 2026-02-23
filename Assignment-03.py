import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# -----------------------------
# Polynomial Function
# -----------------------------
def poly_func(x):
    return 7*(x**4) + 4*(x**3) - x + 6 

# -----------------------------
# Generate Dataset
# -----------------------------
def equation_builder(n_samples=10000):
    # Generate in range [-15, 15]
    x = np.linspace(-15, 15, n_samples)
    np.random.shuffle(x)
    
    # Calculate y values
    y = poly_func(x)
    
    # Normalize input range [-15,15] → [-1,1]
    x_norm = x / 15.0
    
    return x_norm, y

# -----------------------------
# Split Dataset
# -----------------------------
def prepare_data(x, y):
    total_n = len(x)
    
    # Data is already shuffled from equation_builder
    train_percent = int(total_n * 0.8)  # 80% training
    val_percent = int(total_n * 0.1)     # 10% validation
    # 10% test (remaining)
    
    x_train = x[:train_percent]
    y_train = y[:train_percent]
    
    x_val = x[train_percent: train_percent + val_percent]
    y_val = y[train_percent: train_percent + val_percent]
    
    x_test = x[train_percent + val_percent:]
    y_test = y[train_percent + val_percent:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# -----------------------------
# Build DNN (Sequential API)
# -----------------------------
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Linear activation for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# -----------------------------
# Train Model
# -----------------------------
def train_model(model, x_train, y_train, x_val, y_val):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=60,  # Increased for better learning
        verbose=0
    )
    
    return history

# -----------------------------
# Evaluate and Predict
# -----------------------------
def evaluate_and_predict(model, x_test, y_test):
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    
    return loss, mae, predictions, r2

# -----------------------------
# Plot Training Curves
# -----------------------------
def plot_training_curves(history):
    print("\n\n====== Plotting Training Curves =====\n\n")

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    # -------- MSE Plot --------
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Train')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.title("Training vs Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    
    # -------- MAE Plot --------
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mae, 'b-', label='Train')
    plt.plot(epochs_range, val_mae, 'r-', label='Validation')
    plt.title("Training vs Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")

    plt.tight_layout()
    plt.savefig("polynomial_training_curves.png")

# -----------------------------
# Visualize Results
# -----------------------------
def visualize_results(x_test, predictions, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_test, color="blue", label="Actual")
    plt.scatter(x_test, predictions, color='red', label="Predicted")
    plt.title("Polynomial Regression: f(x) = 7x⁴ + 4x³ - x + 6")
    plt.xlabel("Normalized x (range [-1, 1])")
    plt.ylabel("f(x) (original scale)")
    plt.savefig("polynomial_result.png")


# -----------------------------
# Main Function
# -----------------------------
def main():

    # Generate data

    x, y = equation_builder(10000)

    # Split data

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(x, y)
    
    # Build model

    model = build_model()
    model.summary()
    
    # Train model

    history = train_model(model, x_train, y_train, x_val, y_val)
    
    # Evaluate model

    loss, mae, predictions, r2 = evaluate_and_predict(model, x_test, y_test)
    
    print(f"\nTest Results:")
    print(f"   Loss (MSE): {loss:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    visualize_results(x_test, predictions, y_test)
    plot_training_curves(history)
    
    print("\nPlots generated")

if __name__ == "__main__":
    main()
