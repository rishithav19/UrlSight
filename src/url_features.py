import re
import math
import socket
import ipaddress
import pandas as pd
import numpy as np
import tldextract

from urllib.parse import urlparse

# Avoid network requests for the Public Suffix List.
extractor = tldextract.TLDExtract(suffix_list_urls=None)

FEATURE_COLUMNS = [
    "url_len", "dom", "dom_len", "is_ip", "tld", "tld_len",
    "subdom_cnt", "letter_cnt", "digit_cnt", "special_cnt",
    "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt",
    "under_cnt", "letter_ratio", "digit_ratio", "spec_ratio",
    "is_https", "slash_cnt", "entropy", "path_len", "query_len"
]


def calculate_url_features(url):
    """Extract the 24 features used by Dataset.csv."""

    if pd.isna(url):
        raise ValueError("URL cannot be missing.")

    url = str(url).strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    # Parse the URL consistently.
    normalized_url = url if "://" in url else "http://" + url
    parsed = urlparse(normalized_url)

    host = (parsed.hostname or "").lower()

    # Registered domain and public suffix.
    result = extractor(host)

    dom = (
        f"{result.domain}.{result.suffix}"
        if result.domain and result.suffix
        else host
    )

    tld = result.suffix

    subdom_cnt = (
        len(result.subdomain.split("."))
        if result.subdomain else 0
    )

    # Detect IP address hosts.
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # Character counts.
    url_len = len(url)
    letter_cnt = sum(c.isalpha() for c in url)
    digit_cnt = sum(c.isdigit() for c in url)
    special_cnt = sum(not c.isalnum() for c in url)

    eq_cnt = url.count("=")
    qm_cnt = url.count("?")
    amp_cnt = url.count("&")
    dot_cnt = url.count(".")
    dash_cnt = url.count("-")
    under_cnt = url.count("_")

    # Character ratios.
    letter_ratio = letter_cnt / url_len if url_len else 0
    digit_ratio = digit_cnt / url_len if url_len else 0
    spec_ratio = special_cnt / url_len if url_len else 0

    # Shannon entropy of URL characters.
    probabilities = [
        url.count(char) / url_len
        for char in set(url)
    ]

    entropy = -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )

    # Path and query.
    path_len = len(parsed.path)
    query_len = len(parsed.query)

    return {
        "url_len": url_len,
        "dom": dom,
        "dom_len": len(dom),
        "is_ip": is_ip,
        "tld": tld,
        "tld_len": len(tld),
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": eq_cnt,
        "qm_cnt": qm_cnt,
        "amp_cnt": amp_cnt,
        "dot_cnt": dot_cnt,
        "dash_cnt": dash_cnt,
        "under_cnt": under_cnt,
        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "spec_ratio": spec_ratio,
        "is_https": int(parsed.scheme.lower() == "https"),
        "slash_cnt": url.count("/"),
        "entropy": entropy,
        "path_len": path_len,
        "query_len": query_len
    }