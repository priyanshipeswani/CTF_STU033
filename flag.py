import hashlib
import pandas as pd


HASH = "dfd0217d38afa27ccaa06a2b27ff4274a930322e078df5c02e13bd3e80d9e10e"
BOOKS_PATH = "books.csv"
REVIEWS_PATH = "reviews.csv"


MY_HASH = HASH[:8].upper()
print(f" Using hash: {MY_HASH}")

books = pd.read_csv(BOOKS_PATH, on_bad_lines="skip", engine="python")
reviews = pd.read_csv(REVIEWS_PATH, on_bad_lines="skip", engine="python")


target_books = books[
    (books["rating_number"] == 1234) &
    (books["average_rating"] == 5.0)
]

print(f"books found: {len(target_books)}")

mask = reviews["text"].str.contains(MY_HASH, case=False, na=False)

if mask.sum() == 0:
    raise ValueError(" No review found containing your hash!")

fake_review = reviews[mask].iloc[0]

# Correct join column: ASIN
if "asin" not in reviews.columns:
    raise ValueError(" 'asin' column missing in reviews!")

fake_book_asin = fake_review["asin"]
print(f"Fake review ASIN: {fake_book_asin}")


fake_book_code = str(fake_review["asin"])

# Try matching across all possible ID fields
id_columns = ["asin", "parent_asin", "isbn_10", "isbn_13"]

matched_rows = None
matched_column = None

for col in id_columns:
    if col in books.columns:
        temp = books[books[col].astype(str) == fake_book_code]
        if len(temp) > 0:
            matched_rows = temp
            matched_column = col
            break

if matched_rows is None:
    raise ValueError(f" No matching book found for code {fake_book_code} in any ID column.")

print(f" Matched using column: {matched_column}")

my_book = matched_rows.iloc[0]
book_title = my_book["title"]

print(f" Target book title: {book_title}")

title_clean = "".join(book_title.split())[:8]

FLAG1 = hashlib.sha256(title_clean.encode()).hexdigest()
print(f" FLAG1 computed from '{title_clean}':")
print("FLAG1 =", FLAG1)


FLAG2 = f"FLAG2{{{MY_HASH}}}"
print(" FLAG2 =", FLAG2)


with open("flags.txt", "w") as f:
    f.write(f"FLAG1 = {FLAG1}\n")
    f.write(f"FLAG2 = {FLAG2}\n")

print(" Flags written to flags.txt")
