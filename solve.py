# step3_shap.py
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import nltk
import re
import warnings

warnings.filterwarnings("ignore")

BOOKS_PATH = "books.csv"
REVIEWS_PATH = "reviews.csv"


YOUR_HASH = "dfd0217d38afa27ccaa06a2b27ff4274a930322e078df5c02e13bd3e80d9e10e"
MY_HASH = YOUR_HASH[:8].upper()

# Numeric ID appended for FLAG3 (STU160 -> 160)
NUMERIC_ID = "160"

# Text thresholds
SHORT_WORD_TH = 20    
LONG_WORD_TH = 40     

# Words lists - editable
SUPERLATIVES = ["best", "amazing", "awesome", "must-read", "perfect", "incredible", "fantastic", "love", "loved"]
DOMAIN_WORDS = ["characters", "plot", "narrative", "writing", "pacing", "worldbuilding", "prose", "dialogue", "structure"]

# TF-IDF params
MAX_FEATURES = 5000
MIN_DF = 2

# How many low-suspicion reviews to use for SHAP (bottom-k by suspicion score)
N_SHAP_SAMPLES = 50

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t)
    t = t.lower()
    # simple cleanup
    t = re.sub(r"\s+", " ", t).strip()
    return t

def word_count(t):
    return len(t.split())

def contains_any(text, keywords):
    text_l = text.lower()
    for w in keywords:
        if w in text_l:
            return True
    return False

# ---------------------------------------------------
# 1) Load data (tolerant)
# ---------------------------------------------------
print("[*] Loading datasets...")
books = pd.read_csv(BOOKS_PATH, on_bad_lines="skip", engine="python")
reviews = pd.read_csv(REVIEWS_PATH, on_bad_lines="skip", engine="python")

print("[*] Columns in books:", books.columns.tolist())
print("[*] Columns in reviews:", reviews.columns.tolist())

# ---------------------------------------------------
# 2) Find the fake review containing MY_HASH
# ---------------------------------------------------
print(f"[*] Looking for review that contains hash {MY_HASH}...")
mask_hash = reviews["text"].astype(str).str.contains(MY_HASH, case=False, na=False)
if mask_hash.sum() == 0:
    raise SystemExit("No review containing your hash found. Check YOUR_HASH value.")
fake_review = reviews[mask_hash].iloc[0]
print("[+] Found fake review (preview):", fake_review.get("text", "")[:200].replace("\n", " "))

# The identifying code from the review (ASIN)
fake_asin = str(fake_review.get("asin") or fake_review.get("parent_asin") or "")
if not fake_asin:
    raise SystemExit("Fake review does not contain a usable asin/parent_asin field.")

print("[*] Fake review ASIN:", fake_asin)

# ---------------------------------------------------
# 3) Match that code into books to find the exact book
# ---------------------------------------------------
id_columns = ["asin", "parent_asin", "isbn_10", "isbn_13"]
matched_col = None
matched_rows = None
for col in id_columns:
    if col in books.columns:
        tmp = books[books[col].astype(str) == fake_asin]
        if len(tmp) > 0:
            matched_col = col
            matched_rows = tmp
            break

if matched_rows is None:
    # fallback: try partial matches (some datasets use parent_asin)
    for col in id_columns:
        if col in books.columns:
            tmp = books[books[col].astype(str).str.contains(fake_asin, na=False)]
            if len(tmp) > 0:
                matched_col = col
                matched_rows = tmp
                break

if matched_rows is None:
    raise SystemExit(f"Could not find book in books.csv matching code {fake_asin}")

book_row = matched_rows.iloc[0]
book_title = book_row.get("title", "<unknown>")
print(f"[+] Matched book by column '{matched_col}': {book_title}")


def review_matches_book(r):
    for col in id_columns:
        if col in reviews.columns and col in books.columns:
            if str(r.get(col, "")).strip() != "" and str(book_row.get(col, "")).strip() != "":
                if str(r.get(col, "")).strip() == str(book_row.get(col, "")).strip():
                    return True
    # fallback: check asin in review vs book's isbn_10/isbn_13/parent_asin
    if "asin" in reviews.columns:
        rv_asin = str(r.get("asin", "")).strip()
        for bkcol in ["asin", "parent_asin", "isbn_10", "isbn_13"]:
            if bkcol in books.columns:
                if rv_asin != "" and rv_asin == str(book_row.get(bkcol, "")).strip():
                    return True
    return False

print("[*] Subsetting reviews for this book...")
book_reviews = reviews[reviews.apply(review_matches_book, axis=1)].copy()

# If that finds nothing (rare), try matching by asin directly
if len(book_reviews) == 0 and "asin" in reviews.columns:
    book_reviews = reviews[reviews["asin"].astype(str) == fake_asin].copy()

if len(book_reviews) == 0:
    raise SystemExit("No reviews found matching the book. Aborting.")

print(f"[+] Found {len(book_reviews)} reviews for the book")

book_reviews["text_clean"] = book_reviews["text"].apply(clean_text)
book_reviews["word_count"] = book_reviews["text_clean"].apply(word_count)
book_reviews["rating"] = pd.to_numeric(book_reviews["rating"], errors="coerce").fillna(0).astype(int)

# Suspicious rule
book_reviews["has_superlative"] = book_reviews["text_clean"].apply(lambda t: contains_any(t, SUPERLATIVES))
book_reviews["has_domain"] = book_reviews["text_clean"].apply(lambda t: contains_any(t, DOMAIN_WORDS))

def label_row(r):
    
    if r["rating"] == 5 and r["word_count"] < 35 and r["has_superlative"]:
        return 1

    # Genuine (either longer OR domain-focused)
    if r["rating"] == 5 and (r["word_count"] >= 20 or r["has_domain"]):
        return 0

    # ambiguous: return None
    return None

book_reviews["label"] = book_reviews.apply(label_row, axis=1)

# Remove ambiguous samples
labeled = book_reviews[book_reviews["label"].isin([0,1])].copy()
print(f"[+] Labeled reviews: {len(labeled)} (genuine={sum(labeled.label==0)}, suspicious={sum(labeled.label==1)})")

if len(labeled) < 10:
    print("Warning: few labeled samples. Consider loosening thresholds or labeling more data manually.")

# Remove the injected fake review from the training/SHAP set (very important)
fake_mask = labeled["text"].astype(str).str.contains(MY_HASH, case=False, na=False)
if fake_mask.sum() > 0:
    labeled = labeled[~fake_mask]
    print("[+] Removed injected fake review from labeled set")

if len(labeled) < 2:
    raise SystemExit("Not enough labeled samples after removing fake review.")

# ---------------------------------------------------
# 6) Vectorize text and train a classifier
# ---------------------------------------------------
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, ngram_range=(1,2))
X = vectorizer.fit_transform(labeled["text_clean"])
y = labeled["label"].values

# Train-test split (stratify when possible)
if len(np.unique(y)) == 1:
    raise SystemExit("All labeled samples have the same label; cannot train classifier. Adjust labeling rules.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(X_train, y_train)
print("[+] Trained LogisticRegression classifier")
print("[*] Train score:", model.score(X_train, y_train), " Test score:", model.score(X_test, y_test))

# ---------------------------------------------------
# 7) Obtain suspicion scores for all book reviews (including unlabeled)
# ---------------------------------------------------
all_texts = book_reviews["text_clean"].fillna("").values
all_X = vectorizer.transform(all_texts)
# prediction probability of class 1 (suspicious)
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(all_X)[:, 1]
else:
    # fallback: use decision_function mapped to [0,1]
    df = model.decision_function(all_X)
    probs = 1 / (1 + np.exp(-df))

book_reviews = book_reviews.reset_index(drop=True)
book_reviews["suspicion_score"] = probs

# Exclude the injected fake review from SHAP analysis explicitly
book_reviews["is_fake"] = book_reviews["text"].astype(str).str.contains(MY_HASH, case=False, na=False)
print("[*] Reviews marked fake:", book_reviews["is_fake"].sum())

# ---------------------------------------------------
# 8) Select genuine (low suspicion) reviews for SHAP
# ---------------------------------------------------
# We want low suspicion (i.e., low prob of being suspicious)
genuine_candidates = book_reviews[~book_reviews["is_fake"]].sort_values("suspicion_score").head(N_SHAP_SAMPLES)
print(f"[+] Selected {len(genuine_candidates)} low-suspicion reviews for SHAP")

if len(genuine_candidates) == 0:
    raise SystemExit("No genuine candidates for SHAP found — adjust thresholds.")

# Build the matrix for SHAP (use same vectorizer)
shap_X = vectorizer.transform(genuine_candidates["text_clean"].values)

# ---------------------------------------------------
# 9) Run SHAP
# ---------------------------------------------------
print(" Running SHAP... this can take a short while")
# For linear/logistic models, LinearExplainer is appropriate and fast
explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(shap_X)  # shape: (n_samples, n_features)

# shap_values might be returned as list for binary; handle both cases
if isinstance(shap_values, list):
    # shap_values[1] corresponds to class 1 typically
    sv = shap_values[1]
else:
    sv = shap_values

# Aggregate mean SHAP value per feature across selected genuine samples
mean_shap = np.mean(sv, axis=0)  # length = n_features

# We want features that REDUCE suspicion => negative mean SHAP (push toward genuine)
feature_names = vectorizer.get_feature_names_out()
neg_indices = np.argsort(mean_shap)[:50]  # start with 50 most negative candidate features

# Filter to alphanumeric tokens and map back to readable single tokens if needed
top_words = []
for idx in neg_indices:
    token = feature_names[idx]
    # basic token cleanup
    token_clean = re.sub(r"[^a-z0-9\-]", "", token.lower())
    if token_clean == "" or token_clean in top_words:
        continue
    top_words.append(token_clean)
    if len(top_words) >= 3:
        break

if len(top_words) < 3:
    print("Warning: could not find 3 distinct negative tokens; returning what we have:", top_words)

print("[+] Top words that reduce suspicion (candidate):", top_words[:3])

# ---------------------------------------------------
# 10) Build FLAG3
# ---------------------------------------------------
# Concatenate words (lowercase, no spaces) + numeric id
concat = "".join([w.replace(" ", "") for w in top_words[:3]]) + NUMERIC_ID
flag3_hash = hashlib.sha256(concat.encode()).hexdigest()
flag3_short = flag3_hash[:10]
FLAG3 = f"FLAG3{{{flag3_short}}}"

print("[+] Concatenated string for FLAG3:", concat)
print("[+] SHA256:", flag3_hash)
print("[+] FLAG3:", FLAG3)

# ---------------------------------------------------
# 11) Write flags.txt (append or create fresh)
# ---------------------------------------------------
# We try to preserve FLAG1 and FLAG2 if they exist in flags.txt, otherwise we create/update file.
try:
    existing = {}
    try:
        with open("flags.txt", "r") as fh:
            for ln in fh:
                if "=" in ln:
                    k, v = ln.strip().split("=", 1)
                    existing[k.strip()] = v.strip()
    except FileNotFoundError:
        existing = {}

    existing["FLAG3"] = FLAG3

    # if FLAG1 and FLAG2 not present, try compute/store minimal info
    with open("flags.txt", "w") as fh:
        for k in ["FLAG1", "FLAG2", "FLAG3"]:
            if k in existing:
                fh.write(f"{k} = {existing[k]}\n")
            else:
                fh.write(f"# {k} = <missing>\n")
    print("[+] Updated flags.txt with FLAG3")
except Exception as e:
    print("Could not write flags.txt:", e)

# ---------------------------------------------------
# Done
# ---------------------------------------------------
print("[*] Done. If you want, paste the top words here and I'll cross-check/validate the result.")
