from urllib.parse import urlparse


def analyze_url(url, features, prediction, decision_score):
    """Generate transparent rule-based observations from Stage C URL features.

    prediction follows URLSight's normalized convention: 1=phishing,
    0=legitimate. These rules are explanatory heuristics, not model features
    and not proof that a URL is malicious or safe.
    """
    parsed = urlparse(url)
    indicators = []
    reassuring_signals = []

    if features.get("has_ip_address") == 1:
        indicators.append("The hostname is a raw IP address rather than a domain name.")

    if features.get("subdomain_count", 0) >= 3:
        indicators.append(
            "The URL contains several subdomain levels. Check which domain is actually registered."
        )

    if features.get("url_length", 0) > 100:
        indicators.append(
            "The URL is unusually long, which can make its destination harder to inspect."
        )

    if features.get("percentage_numeric_chars", 0) > 25:
        indicators.append("Digits make up a relatively large proportion of the URL.")

    if features.get("url_entropy", 0) > 4.2:
        indicators.append(
            "The URL has relatively high character entropy, which can occur in obfuscated or randomly generated strings."
        )

    if features.get("query_param_count", 1) > 4:
        indicators.append("The URL contains several query parameters.")

    if features.get("suspicious_file_extension") == 1:
        indicators.append(
            "The URL path ends with a file extension commonly associated with executable or script downloads."
        )

    if features.get("has_hyphen_in_domain") == 1:
        indicators.append("The domain label contains a hyphen; this is worth checking in context.")

    if features.get("https_flag") == 1:
        reassuring_signals.append(
            "The URL uses HTTPS, which encrypts the connection but does not by itself establish that the website is trustworthy."
        )
    else:
        indicators.append("The URL does not use HTTPS.")

    if features.get("tld_popularity") == 1:
        reassuring_signals.append(
            "The top-level domain belongs to the model's common-TLD set; this alone does not establish trustworthiness."
        )

    if not indicators:
        indicators.append(
            "No suspicious indicators were detected by these simple URL-structure checks. This does not guarantee safety."
        )

    if prediction == 1:
        verdict = "The Stage C URL model classified this URL as phishing."
        recommendation = (
            "Avoid entering credentials or payment information. Verify the domain through an independent, trusted source."
        )
    else:
        verdict = "The Stage C URL model classified this URL as legitimate."
        recommendation = (
            "This is not a safety guarantee. Check the domain carefully before entering sensitive information."
        )

    return {
        "verdict": verdict,
        "indicators": indicators,
        "reassuring_signals": reassuring_signals,
        "recommendation": recommendation,
        "decision_score": float(decision_score),
        "decision_score_note": "Positive values favor phishing; negative values favor legitimate. This is not a probability.",
    }
