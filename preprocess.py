# services/preprocess.py
import re
from typing import Tuple, List

def extract_entities_and_keywords(text: str) -> Tuple[List[str], List[str]]:
    """
    Very simple heuristic extraction:
      - Entities: capitalized words (quick)
      - Keywords: nouns/verbs via regex split
    Replace with spaCy / transformers for production.
    """
    # entity heuristic (very naive)
    entities = re.findall(r'\b[A-Z][a-zA-Z0-9&.-]{1,}\b', text)
    # keywords: remove stopwords for now (simple)
    tokens = re.findall(r"\w+", text.lower())
    stopwords = {"the","is","in","on","at","a","an","and","or","of","for","to","from","by"}
    keywords = [t for t in tokens if t not in stopwords]
    return list(set(entities)), list(dict.fromkeys(keywords))  # unique, preserve order
