import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Training data (Hours studied -> Scores)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Define a simple neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=[1]),
    keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=500, verbose=0)

# Ask user for input
hours = float(input("Enter hours studied: "))

# Predict
pred = model.predict(np.array([[hours]]))
print(f"Prediction for studying {hours} hours: {pred[0][0]}")

# Plot training data, regression curve, and user input
plt.scatter(X, y, color="blue", label="Training data")
plt.scatter([hours], [pred[0][0]], color="green", marker="x", s=100, label="Your input")

# Plot the model's regression line (or curve)
X_line = np.linspace(0, 10, 100)  # smoother line
X_line_reshaped = X_line.reshape(-1, 1)  # reshape to 2D
y_line = model.predict(X_line_reshaped)

plt.plot(X_line, y_line, color="red", label="Neural Network Fit")

plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.legend()
plt.title("Deep Learning Linear Regression")
plt.show()
