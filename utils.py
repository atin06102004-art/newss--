"""
utils.py
Text preprocessing utilities: cleaning, stopword removal, lemmatization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure required NLTK data is available.
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/numbers/URLs, remove stopwords, and
    lemmatize. Returns a single cleaned string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        _LEMMATIZER.lemmatize(word)
        for word in text.split()
        if word not in _STOPWORDS and len(word) > 2
    ]
    return " ".join(tokens)
