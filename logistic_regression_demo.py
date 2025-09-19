import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Example dataset (Hours studied -> Pass/Fail)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)  # 0 = Fail, 1 = Pass

# Define a logistic regression model (neural net with sigmoid)
model = keras.Sequential([
    keras.layers.Dense(1, activation="sigmoid", input_shape=[1])
])

# Compile with binary crossentropy (since it's binary classification)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(X, y, epochs=500, verbose=0)

# Predictions
preds = model.predict(np.array([[3], [5], [7]]))
print("Predictions (probabilities):", preds.flatten())
print("Class labels:", (preds > 0.5).astype(int).flatten())

# Plot training accuracy
plt.plot(history.history['accuracy'], label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.show()
