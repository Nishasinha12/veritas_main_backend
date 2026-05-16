# services/scorer.py
from typing import List, Dict
import math

def compute_truth_score(claim: str, news_results: List[Dict], social_results: List[Dict]) -> Dict:
    """
    Simple heuristic scorer:
     - Weighted average of source credibility (news)
     - Penalize if more social spread but low credible news support
     - Use textual similarity (not implemented here) to check if articles support/deny claim.
    Replace scoring with ML model / rule engine as you improve.
    """
    # news credibility average
    if news_results:
        cred_scores = [r.get("credibility_score", 0.5) for r in news_results]
        news_score = sum(cred_scores) / len(cred_scores)
    else:
        news_score = 0.5

    # social signal strength (normalize)
    social_signal = 0.0
    if social_results:
        total_engagement = 0
        for s in social_results:
            m = s.get("metrics", {})
            total_engagement += m.get("retweets", 0) + m.get("likes", 0) + m.get("replies",0)
        # log-scale normalize
        social_signal = min(1.0, math.log1p(total_engagement) / 10.0)

    # naive truth score: news_score adjusted downward if social signal high but news low
    # This is heuristic — replace later with semantic claim-evidence analysis
    truth_score = news_score * (1 - 0.4*social_signal) + 0.1*social_signal
    truth_score = max(0.0, min(1.0, truth_score))

    # verdict thresholds
    if truth_score >= 0.75:
        verdict = "Likely True"
    elif truth_score <= 0.35:
        verdict = "Likely False"
    else:
        verdict = "Unverified / Mixed"

    return {
        "verdict": verdict,
        "score": round(truth_score, 3),
        "sources": news_results,
        "social_context": social_results
    }
