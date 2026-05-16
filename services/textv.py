# services/textv.py
# Stub module for text verification key loading
# This is used by verify_api/entrypoint.py

_NEWS_API_KEY = None
_GNEWS_API_KEY = None
_MEDIASTACK_API_KEY = None

def load_text_keys(news_api_key, gnews_api_key, mediastack_api_key):
    """Load API keys for text verification services."""
    global _NEWS_API_KEY, _GNEWS_API_KEY, _MEDIASTACK_API_KEY
    _NEWS_API_KEY = news_api_key
    _GNEWS_API_KEY = gnews_api_key
    _MEDIASTACK_API_KEY = mediastack_api_key
