import hashlib

def sha256_hash(text: str) -> str:
    # Encode text to bytes and compute SHA-256 hash
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

if __name__ == "__main__":
    user_input = input("Enter text to hash: ")
    print("SHA-256 Hash:")
    print(sha256_hash(user_input))
