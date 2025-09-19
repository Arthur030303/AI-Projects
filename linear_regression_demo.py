import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Training data (X = hours studied, y = scores)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Define a simple neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=[1]),
    keras.layers.Dense(1)  # output layer
])

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(X, y, epochs=500, verbose=0)

# Predict
pred = model.predict([6.0])
print("Prediction for studying 6 hours:", pred[0][0])

# Plot loss curve
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
