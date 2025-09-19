import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Training data (Hours studied -> Scores)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Ask user for input
hours = float(input("Enter hours studied: "))

# Predict
pred = model.predict([[hours]])
print(f"Prediction for studying {hours} hours: {pred[0]}")

# Plot regression line and user input
plt.scatter(X, y, color="blue", label="Training data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.scatter([hours], [pred[0]], color="green", marker="x", s=100, label="Your input")
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.legend()
plt.title("Linear Regression Demo")
plt.show()
