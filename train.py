import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

print("Loading data...")
df = pd.read_csv("complaints.csv", low_memory=False)

df = df[["Consumer complaint narrative", "Product"]].dropna()
df.columns = ["text", "label"]

print(f"Total complaints with text: {len(df)}")

top5 = df["label"].value_counts().head(5).index.tolist()
df = df[df["label"].isin(top5)]

print("Categories we are classifying:")
for i, cat in enumerate(top5):
    count = len(df[df["label"] == cat])
    print(f"  {i+1}. {cat} ({count} complaints)")

df = df.sample(n=min(50000, len(df)), random_state=42)
print(f"\nUsing {len(df)} complaints for training")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42
)

print(f"Training on {len(X_train)} complaints")
print(f"Testing on {len(X_test)} complaints")

print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

print("\nResults:")
predictions = model.predict(X_test_vec)
print(classification_report(y_test, predictions))

print("Saving model...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nDone! model.pkl and vectorizer.pkl saved.")
print("Run: python3 -m streamlit run app.py")
