import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras

# === Step 1: Training Data ===
training_sentences = [
    "hello", "hi", "good morning",
    "bye", "see you", "goodnight",
    "thanks", "thank you", "much appreciated",
    "what's the weather", "is it hot today", "will it rain"
]

training_labels = [
    "greet", "greet", "greet",
    "bye", "bye", "bye",
    "thanks", "thanks", "thanks",
    "weather", "weather", "weather"
]

label_classes = list(set(training_labels))

# === Step 2: Vectorize Text ===
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

# Encode labels as numbers
label_to_id = {label: idx for idx, label in enumerate(label_classes)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
y = np.array([label_to_id[label] for label in training_labels])

# === Step 3: Machine Learning Model (Logistic Regression) ===
ml_model = LogisticRegression()
ml_model.fit(X, y)

# === Step 4: Deep Learning Model (Neural Network) ===
dl_model = keras.Sequential([
    keras.layers.Dense(8, activation="relu", input_shape=(X.shape[1],)),
    keras.layers.Dense(len(label_classes), activation="softmax")
])

dl_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
dl_model.fit(X.toarray(), y, epochs=200, verbose=0)

# === Step 5: Chatbot Function ===
def chatbot_response(user_input):
    # Transform input
    X_input = vectorizer.transform([user_input])

    # ML Prediction
    ml_pred = ml_model.predict(X_input)[0]
    ml_intent = id_to_label[ml_pred]

    # DL Prediction
    dl_pred = dl_model.predict(X_input.toarray())
    dl_intent = id_to_label[np.argmax(dl_pred)]

    # Responses
    responses = {
        "greet": "Hello! How can I help you?",
        "bye": "Goodbye! Have a nice day!",
        "thanks": "You're welcome!",
        "weather": "I think it’s sunny today!"
    }

    return f"[ML] {responses[ml_intent]}\n[DL] {responses[dl_intent]}"

# === Step 6: Chat Loop ===
print("Chatbot is running! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print(chatbot_response(user_input))
