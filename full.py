import pandas as pd
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import nltk

# Make sure punkt_tab tokenizer is installed
nltk.download('punkt_tab')

# --------------------------------------------
# Helper functions
# --------------------------------------------

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def take_first_8_nonspace(text: str) -> str:
    return "".join(text.split())[:8]

# --------------------------------------------
# Step 0: Your personal hash from "STU160"
# --------------------------------------------

YOUR_ID = "STU033"
your_hash = sha256_hex(YOUR_ID)[:8].upper()
print("Your HASH =", your_hash)


books = pd.read_csv("books.csv")
reviews = pd.read_csv("reviews.csv")

# --------------------------------------------
# Step 1: Find your book
# --------------------------------------------

sus_books = books[
    (books["rating_number"] == 1234) &
    (books["average_rating"] == 5.0)
]

# Merge reviews with these books
merged = reviews.merge(sus_books, on="book_id", suffixes=("_review", "_book"))

# Find review containing your hash
fake_review_row = merged[merged["text"].str.contains(your_hash, case=False, na=False)].iloc[0]

target_book_id = fake_review_row["book_id"]
target_title = fake_review_row["title"]

print("Target Book ID =", target_book_id)
print("Target Title =", target_title)

# Compute FLAG1
title_8 = take_first_8_nonspace(target_title)
FLAG1 = sha256_hex(title_8)

print("FLAG1 =", FLAG1)

# --------------------------------------------
# Step 2: FLAG2 (fake review)
# --------------------------------------------

FLAG2 = f"FLAG2{{{your_hash}}}"
print("FLAG2 =", FLAG2)

# --------------------------------------------
# Step 3: Authenticity Model + SHAP
# --------------------------------------------

# Extract all reviews for this book
book_reviews = reviews[reviews["book_id"] == target_book_id].copy()

# Label suspicious vs genuine based on rules
def is_suspicious(text, stars):
    text_len = len(text.split())
    if stars == 5 and text_len < 12:  # short 5-star
        return 1
    # add superlatives rule
    if any(w in text.lower() for w in ["amazing", "best", "incredible", "perfect"]):
        return 1
    return 0

book_reviews["label"] = book_reviews.apply(
    lambda r: is_suspicious(r["text"], r["rating"]),
    axis=1
)

# Vectorize
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(book_reviews["text"])
y = book_reviews["label"]

# Train classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# Predict suspicion
book_reviews["prob"] = clf.predict_proba(X)[:, 1]

# Keep genuine (low suspicion)
genuine_idx = book_reviews["prob"].nsmallest(10).index
X_genuine = X[genuine_idx]

# SHAP on genuine reviews
explainer = shap.LinearExplainer(clf, X, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_genuine)

# Get average SHAP effect per word
mean_shap = shap_values.mean(axis=0)

# Words that reduce suspicion (most negative SHAP)
top3_indices = mean_shap.argsort()[:3]
words = [tfidf.get_feature_names_out()[i] for i in top3_indices]

print("Top 3 SHAP helpful words:", words)

# --------------------------------------------
# Compute FLAG3
# --------------------------------------------

concat = "".join(words) + YOUR_ID[3:]  # numeric ID "160"
FLAG3_raw = sha256_hex(concat)[:10]
FLAG3 = f"FLAG3{{{FLAG3_raw}}}"

print("FLAG3 =", FLAG3)

# --------------------------------------------
# Write flags.txt
# --------------------------------------------

with open("flags.txt", "w") as f:
    f.write(f"FLAG1 = {FLAG1}\n")
    f.write(f"{FLAG2}\n")
    f.write(f"{FLAG3}\n")

print("\nAll flags saved to flags.txt")
