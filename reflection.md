# Reflection on CTF Challenge Methodology

## Approach Overview
My approach to this CTF challenge involved systematic data analysis and machine learning techniques to uncover the manipulated book review.

## Step-by-Step Process

**FLAG1 Discovery**: I computed the SHA256 hash of "STU033" and searched for books with rating_number=1234 and average_rating=5.0. By scanning reviews for the hash signature, I identified the target book and extracted the first 8 non-space characters from its title to generate FLAG1.

**FLAG2 Identification**: Located the specific review containing my hash signature (DFD0217D), which directly provided FLAG2.

**FLAG3 SHAP Analysis**: The most complex step involved training a binary classifier on the entire review dataset to distinguish suspicious reviews (short, superlative-heavy, 5-star) from genuine ones (longer, domain-specific language). I used TF-IDF vectorization and logistic regression, then applied SHAP analysis to genuine reviews (those with low suspicion scores) to identify the top 3 words that most strongly indicate authenticity. These words were concatenated with my numeric ID (160) and hashed to produce FLAG3.

The key insight was using the full dataset for training rather than just the target book's reviews, enabling robust pattern recognition.