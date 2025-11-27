#!/usr/bin/env python3
"""
FLAG3 Solution: SHAP Analysis for Authentic Review Detection
============================================================

Steps:
1. Load all reviews from the dataset
2. Label reviews as suspicious vs genuine based on patterns
3. Train a classifier on all reviews (not just the target book)
4. Use SHAP to find words that make reviews appear MORE GENUINE
5. Extract FLAG3 from top 3 authenticity-indicating words
"""

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

# Configuration
BOOKS_PATH = "books.csv"
REVIEWS_PATH = "reviews.csv"
YOUR_HASH = "DFD0217D"  # From FLAG2
NUMERIC_ID = "160"  # STU160 -> 160

# Classification thresholds
SHORT_WORD_THRESHOLD = 20
LONG_WORD_THRESHOLD = 50

# Keywords for classification
SUPERLATIVES = [
    "best", "amazing", "awesome", "must-read", "perfect", "incredible", 
    "fantastic", "love", "loved", "great", "excellent", "outstanding",
    "wonderful", "brilliant", "masterpiece"
]

DOMAIN_WORDS = [
    "characters", "plot", "narrative", "writing", "pacing", "worldbuilding",
    "prose", "dialogue", "structure", "development", "storyline", "theme",
    "character", "author", "chapter", "story", "book", "novel", "fiction"
]

# TF-IDF settings
MAX_FEATURES = 5000
MIN_DF = 3
N_SHAP_SAMPLES = 100

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def word_count(text):
    """Count words in text"""
    return len(text.split()) if text else 0

def contains_keywords(text, keywords):
    """Check if text contains any of the keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

def classify_review(row):
    """
    Classify review as suspicious (1) or genuine (0)
    
    Suspicious: 5-star + short + superlatives
    Genuine: 5-star + longer OR contains domain-specific terms
    """
    if row['rating'] != 5:
        return None  # Only classify 5-star reviews
    
    word_cnt = row['word_count']
    has_superlatives = row['has_superlatives']
    has_domain = row['has_domain']
    
    # Suspicious: short + superlative-heavy
    if word_cnt < SHORT_WORD_THRESHOLD and has_superlatives:
        return 1
    
    # Genuine: longer reviews OR domain-focused
    if word_cnt >= SHORT_WORD_THRESHOLD or has_domain:
        return 0
    
    return None  # Ambiguous cases

def main():
    print("Loading dataset...")
    
    # Load data
    reviews = pd.read_csv(REVIEWS_PATH, on_bad_lines='skip', engine='python')
    print(f"Loaded {len(reviews):,} reviews")
    
    # Clean and prepare data
    print("Preprocessing text...")
    reviews['text_clean'] = reviews['text'].apply(clean_text)
    reviews['word_count'] = reviews['text_clean'].apply(word_count)
    reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce').fillna(0)
    
    # Filter to 5-star reviews only
    five_star = reviews[reviews['rating'] == 5].copy()
    print(f"Found {len(five_star):,} five-star reviews")
    
    # Feature engineering for classification
    five_star['has_superlatives'] = five_star['text_clean'].apply(
        lambda x: contains_keywords(x, SUPERLATIVES)
    )
    five_star['has_domain'] = five_star['text_clean'].apply(
        lambda x: contains_keywords(x, DOMAIN_WORDS)
    )
    
    # Classify reviews
    five_star['label'] = five_star.apply(classify_review, axis=1)
    labeled_reviews = five_star[five_star['label'].isin([0, 1])].copy()
    
    genuine_count = (labeled_reviews['label'] == 0).sum()
    suspicious_count = (labeled_reviews['label'] == 1).sum()
    
    print(f"Classified reviews:")
    print(f"   Genuine: {genuine_count:,}")
    print(f"   Suspicious: {suspicious_count:,}")
    
    if len(labeled_reviews) < 100:
        raise ValueError("Not enough labeled samples for training")
    
    # Remove the injected fake review if it exists
    fake_mask = labeled_reviews['text'].astype(str).str.contains(YOUR_HASH, case=False, na=False)
    if fake_mask.sum() > 0:
        labeled_reviews = labeled_reviews[~fake_mask]
        print(f"Removed {fake_mask.sum()} fake review(s)")
    
    # Prepare training data
    print("Training classifier...")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(labeled_reviews['text_clean'])
    y = labeled_reviews['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Model trained - Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")
    
    # Get suspicion scores for all reviews
    print("Computing suspicion scores...")
    all_X = vectorizer.transform(labeled_reviews['text_clean'])
    suspicion_scores = model.predict_proba(all_X)[:, 1]  # Probability of being suspicious
    
    # Select genuine reviews (low suspicion scores) for SHAP
    labeled_reviews = labeled_reviews.reset_index(drop=True)
    labeled_reviews['suspicion_score'] = suspicion_scores
    
    # Get the most genuine reviews (lowest suspicion scores)
    genuine_reviews = labeled_reviews.nsmallest(N_SHAP_SAMPLES, 'suspicion_score')
    print(f"Selected {len(genuine_reviews)} most genuine reviews for SHAP analysis")
    
    # SHAP analysis
    print("Running SHAP analysis...")
    shap_X = vectorizer.transform(genuine_reviews['text_clean'])
    
    # Use LinearExplainer for logistic regression
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(shap_X)
    
    # Handle SHAP output format
    if isinstance(shap_values, list):
        # For binary classification, take values for suspicious class (class 1)
        sv = shap_values[1] 
    else:
        sv = shap_values
    
    # Calculate mean SHAP values across all genuine reviews
    mean_shap_values = np.mean(sv, axis=0)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Find features that REDUCE suspicion (most negative SHAP values)
    # These are words that make reviews appear more genuine
    most_negative_indices = np.argsort(mean_shap_values)
    
    print("Top words that indicate authenticity (reduce suspicion):")
    
    top_authentic_words = []
    for idx in most_negative_indices:
        feature = feature_names[idx]
        shap_val = mean_shap_values[idx]
        
        # Clean the feature to get readable words
        clean_feature = re.sub(r'[^a-z0-9]', '', feature.lower())
        
        if (len(clean_feature) >= 3 and 
            clean_feature.isalpha() and 
            clean_feature not in top_authentic_words):
            
            top_authentic_words.append(clean_feature)
            print(f"   {len(top_authentic_words)}. '{clean_feature}' (SHAP: {shap_val:.4f})")
            
            if len(top_authentic_words) >= 3:
                break
    
    if len(top_authentic_words) < 3:
        print("Warning: Could not find 3 distinct authentic words")
        # Pad with generic words if needed
        generic_words = ["book", "story", "read"]
        while len(top_authentic_words) < 3:
            word = generic_words[len(top_authentic_words) - len([w for w in top_authentic_words if w in generic_words])]
            if word not in top_authentic_words:
                top_authentic_words.append(word)
    
    # Generate FLAG3
    print(f"\nGenerating FLAG3...")
    
    # Concatenate top 3 words + numeric ID
    flag3_string = "".join(top_authentic_words[:3]) + NUMERIC_ID
    print(f"   Concatenated string: '{flag3_string}'")
    
    # Compute SHA256 and take first 10 characters
    flag3_hash = hashlib.sha256(flag3_string.encode()).hexdigest()
    flag3_result = flag3_hash[:10]
    FLAG3 = f"FLAG3{{{flag3_result}}}"
    
    print(f"   SHA256: {flag3_hash}")
    print(f"   FLAG3: {FLAG3}")
    
    # Update flags.txt
    print("\nUpdating flags.txt...")
    
    # Read existing flags
    existing_flags = {}
    try:
        with open("flags.txt", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    existing_flags[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    
    # Add FLAG3
    existing_flags["FLAG3"] = FLAG3
    
    # Write all flags
    with open("flags.txt", "w") as f:
        for flag_name in ["FLAG1", "FLAG2", "FLAG3"]:
            if flag_name in existing_flags:
                f.write(f"{flag_name} = {existing_flags[flag_name]}\n")
            else:
                f.write(f"# {flag_name} = <not found>\n")
    
    print("flags.txt updated successfully!")
    print(f"\nFLAG3 completed: {FLAG3}")
    
    return FLAG3

if __name__ == "__main__":
    try:
        flag3 = main()
        print(f"\nSuccess! Your FLAG3 is: {flag3}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()