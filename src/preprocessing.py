import re

def clean_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing HTML tags
    - Removing non-alphabetic characters (punctuation and numbers)
    - Stripping extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataframe(df):
    """
    Applies the clean_text function to the combined_text column.
    """
    print("Cleaning text data. This might take a moment...")
    df["clean_text"] = df["combined_text"].apply(clean_text)
    return df
