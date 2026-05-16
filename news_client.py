import requests
import os
import urllib.parse  # ✅ add this

API_KEY = os.getenv("NEWS_API_KEY")

def query_news(claim):
    query_encoded = urllib.parse.quote(claim)  # ✅ encode spaces
    url = f"https://newsapi.org/v2/everything?q={query_encoded}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = []
    for article in data.get("articles", []):
        articles.append({
            "title": article.get("title"),
            "url": article.get("url"),
            "source": article.get("source", {}).get("name"),
            "source_domain": article.get("source", {}).get("id"),
            "description": article.get("description"),
            "published_at": article.get("publishedAt"),
            "country": article.get("source", {}).get("id", "unknown"),
            "credibility_score": get_credibility(article.get("source", {}).get("name", ""))
        })
    return articles

CREDIBILITY_MAP = {
    "BBC News": 0.95,
    "Reuters": 0.95,
    "The Guardian": 0.90,
    "Associated Press": 0.93,
    "CNN": 0.75,
    "Fox News": 0.60,
    "The New York Times": 0.88,
}

def get_credibility(source_name: str) -> float:
    return CREDIBILITY_MAP.get(source_name, 0.5)
