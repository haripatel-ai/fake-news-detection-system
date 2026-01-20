---

### 🧠 `fake_news.py`
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# User input
while True:
    news = input("\nEnter news text (or 'exit'): ")
    if news.lower() == "exit":
        break
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)
    print("📰 Result:", "FAKE NEWS" if prediction[0] == 0 else "REAL NEWS")