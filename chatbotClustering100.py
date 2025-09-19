from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 100 chatbot-like example sentences
sentences = [
    # Greetings
    "hello", "hi", "hey there", "good morning", "good afternoon", "good evening",
    "how are you", "how's it going", "what's up", "nice to meet you",
    "long time no see", "how have you been", "it's good to see you", "yo", "hiya",

    # Weather
    "what is the weather", "is it sunny today", "will it rain", "how's the weather outside",
    "is it cloudy", "do I need an umbrella", "what’s the temperature",
    "is it hot today", "is it cold today", "is there a storm coming",

    # Time / Date
    "what time is it", "can you tell me the time", "what day is it today",
    "what's the date", "is it morning", "is it night", "what month is it",
    "what year is it", "what’s today’s date", "when is new year",

    # Jokes / Fun
    "tell me a joke", "make me laugh", "say something funny", "can you cheer me up",
    "do you know a joke", "tell me something cool", "make a pun", "say something silly",
    "do you like jokes", "make me smile",

    # Small talk
    "what are you doing", "do you like music", "what is your favorite color",
    "do you eat food", "can you dance", "do you play games", "do you like movies",
    "what is your hobby", "do you sleep", "do you dream",

    # Thanks
    "thank you", "thanks", "thanks a lot", "much appreciated", "cheers",
    "thanks so much", "thank you very much", "many thanks", "thanks buddy", "grateful",

    # Farewells
    "bye", "goodbye", "see you later", "catch you later", "talk to you soon",
    "see you tomorrow", "take care", "farewell", "see you around", "later",

    # Yes/No responses
    "yes", "no", "maybe", "of course", "definitely", "not really",
    "absolutely", "I don’t think so", "sure", "nope",

    # Help / Requests
    "can you help me", "I need help", "what can you do", "show me something",
    "give me information", "how can you help", "assist me please",
    "what services do you offer", "help me out", "can you explain",

    # Random
    "who are you", "what are you", "are you real", "do you have feelings",
    "do you think", "are you alive", "do you understand me", "do you like humans",
    "can you learn", "are you smart"
]

# Step 1: Convert text to features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sentences)

# Step 2: Cluster
k = 8  # we can try 8 clusters (greetings, weather, time, jokes, small talk, thanks, farewells, misc)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)

# Step 3: Show results
print("\n=== Chatbot Sentence Clusters ===")
for i, sentence in enumerate(sentences):
    print(f"Cluster {kmeans.labels_[i]} → {sentence}")
