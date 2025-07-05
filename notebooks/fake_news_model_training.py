import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Load datasets
fake = pd.read_csv('../data/Fake.csv')
real = pd.read_csv('../data/True.csv')

# Add labels
fake['label'] = 0  # Fake
real['label'] = 1  # Real

# Combine & shuffle
data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)

# Features and Labels
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model: Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model & vectorizer
os.makedirs('../model', exist_ok=True)
joblib.dump(model, '../model/news_model.pkl')
joblib.dump(vectorizer, '../model/vectorizer.pkl')
print("Model and vectorizer saved successfully.")
