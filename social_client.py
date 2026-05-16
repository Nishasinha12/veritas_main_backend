# services/social_client.py
from typing import List, Dict
import requests

def query_social(keywords: List[str], entities: List[str]) -> List[Dict]:
    """
    Fetch recent social media posts relevant to the claim.
    Replace the TODO sections with real API calls:
        - Twitter/X API
        - Reddit API
        - Facebook Graph API
    Returns a list of dictionaries with keys:
        'platform', 'url', 'text', 'user', 'metrics', 'timestamp'
    """
    query = " ".join(keywords + entities)

    posts: List[Dict] = []

    # -------------------------------
    # TODO: Twitter/X API integration
    # Example:
    # response = requests.get(TWITTER_API_URL, params={"query": query}, headers=HEADERS)
    # tweets = response.json().get("data", [])
    # for t in tweets:
    #     posts.append({
    #         "platform": "Twitter",
    #         "url": t["url"],
    #         "text": t["text"],
    #         "user": t["author_username"],
    #         "metrics": {"likes": t["like_count"], "retweets": t["retweet_count"]},
    #         "timestamp": t["created_at"]
    #     })
    # -------------------------------

    # -------------------------------
    # TODO: Reddit API integration
    # -------------------------------

    # -------------------------------
    # TODO: Facebook Graph API integration
    # -------------------------------

    return posts  # Empty list if no posts found
