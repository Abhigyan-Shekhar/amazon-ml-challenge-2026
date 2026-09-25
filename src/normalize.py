import re
import unicodedata

VERSION = "unicode-nfkc-lower-punctuation-space-v1"

def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", " ", "".join(c if c.isalnum() or c.isspace() else " " for c in value)).strip()

def prepare(frame):
    frame = frame.copy()
    frame["raw_name"] = frame.business_name
    frame["raw_address"] = frame.business_address
    frame["normalized_name"] = frame.business_name.map(normalize)
    frame["normalized_address"] = frame.business_address.map(normalize)
    frame["normalized_country"] = frame.country.map(normalize)
    frame["retrieval_text"] = frame.normalized_name + " " + frame.normalized_address
    frame["combined_text"] = "[NAME] " + frame.normalized_name + " [ADDRESS] " + frame.normalized_address + " [COUNTRY] " + frame.normalized_country
    return frame
