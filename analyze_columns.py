import pandas as pd

# Load CSV files
books = pd.read_csv("books.csv", on_bad_lines="skip", engine="python")
reviews = pd.read_csv("reviews.csv", on_bad_lines="skip", engine="python")

# Print column names
print("Books columns:", books.columns.tolist())
print("Reviews columns:", reviews.columns.tolist())

# Identify common columns
common_columns = list(set(books.columns).intersection(set(reviews.columns)))
print("Common columns:", common_columns)
