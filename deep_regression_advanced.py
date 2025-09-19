import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Example dataset (Hours studied -> Scores)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=float)

# Define a deeper neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=[1]),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)  # output layer
])

# Compile model with a smaller learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Use EarlyStopping callback to avoid overtraining
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

# Train with validation split
history = model.fit(X, y, epochs=500, validation_split=0.2, callbacks=[early_stop], verbose=0)

# Predict
pred = model.predict(np.array([[10.0]]))
print("Prediction for studying 10 hours:", pred[0][0])

# Plot training vs validation loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
