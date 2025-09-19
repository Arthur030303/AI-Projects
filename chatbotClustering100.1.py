from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 100 chatbot-like example sentences
sentences = [
    "hello", "hi", "hey there", "good morning", "good afternoon", "good evening",
    "how are you", "how's it going", "what's up", "nice to meet you",
    "long time no see", "how have you been", "it's good to see you", "yo", "hiya",
    "what is the weather", "is it sunny today", "will it rain", "how's the weather outside",
    "is it cloudy", "do I need an umbrella", "what’s the temperature",
    "is it hot today", "is it cold today", "is there a storm coming",
    "what time is it", "can you tell me the time", "what day is it today",
    "what's the date", "is it morning", "is it night", "what month is it",
    "what year is it", "what’s today’s date", "when is new year",
    "tell me a joke", "make me laugh", "say something funny", "can you cheer me up",
    "do you know a joke", "tell me something cool", "make a pun", "say something silly",
    "do you like jokes", "make me smile",
    "what are you doing", "do you like music", "what is your favorite color",
    "do you eat food", "can you dance", "do you play games", "do you like movies",
    "what is your hobby", "do you sleep", "do you dream",
    "thank you", "thanks", "thanks a lot", "much appreciated", "cheers",
    "thanks so much", "thank you very much", "many thanks", "thanks buddy", "grateful",
    "bye", "goodbye", "see you later", "catch you later", "talk to you soon",
    "see you tomorrow", "take care", "farewell", "see you around", "later",
    "yes", "no", "maybe", "of course", "definitely", "not really",
    "absolutely", "I don’t think so", "sure", "nope",
    "can you help me", "I need help", "what can you do", "show me something",
    "give me information", "how can you help", "assist me please",
    "what services do you offer", "help me out", "can you explain",
    "who are you", "what are you", "are you real", "do you have feelings",
    "do you think", "are you alive", "do you understand me", "do you like humans",
    "can you learn", "are you smart"
]

# Convert text to features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)

# Cluster
k = 8
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)

# Show results grouped by cluster
clusters = {i: [] for i in range(k)}
for i, label in enumerate(kmeans.labels_):
    clusters[label].append(sentences[i])

print("\n=== Chatbot Sentence Clusters ===")
for cluster_id, cluster_sentences in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for s in cluster_sentences:
        print(f"  - {s}")

# Optional: let user test their own input
while True:
    user_input = input("\nType a message (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    X_user = vectorizer.transform([user_input])
    cluster = kmeans.predict(X_user)[0]
    print(f"Your input belongs to Cluster {cluster}")
