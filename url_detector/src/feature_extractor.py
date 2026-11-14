r"""
Inference Feature Extractor for Phishing URL Detection
Stage C Component: Exact feature extractor matching LegitPhish dataset definitions.

Features Extracted (16 numerical features):
1.  url_length: Total character length of URL
2.  has_ip_address: Binary flag indicating if domain is a raw IPv4/IPv6 address
3.  dot_count: Count of '.' characters in the entire URL
4.  https_flag: Binary flag (1 if scheme is https, 0 otherwise)
5.  url_entropy: Shannon entropy of character distribution in the URL
6.  token_count: Number of tokens split by delimiters [/.\?#=]
7.  subdomain_count: Number of subdomains (len(domain_parts) - 2)
8.  query_param_count: len(urlparse(url).query.split('&')), preserves dataset offset where no-query URLs produce 1
9.  tld_length: Character length of TLD (netloc.split('.')[-1] if len >= 2 else 0)
10. path_length: Character length of URL path component (urlparse(url).path)
11. has_hyphen_in_domain: Binary flag if '-' is present in domain name
12. number_of_digits: Count of numeric digits in entire URL
13. tld_popularity: Binary flag (1 if TLD in {'com', 'org', 'net', 'edu', 'gov'}, else 0)
14. suspicious_file_extension: Binary flag (1 if path ends in .exe, .zip, .bin, .apk, .cmd, .js, else 0)
15. domain_name_length: Character length of registered domain label (netloc.split('.')[-2] if len >= 2 else 0)
16. percentage_numeric_chars: (number_of_digits / url_length) * 100.0
"""

import math
import re
import os
import ipaddress
from urllib.parse import urlparse
from collections import Counter
from typing import Dict, Any, List

POPULAR_TLDS = {'com', 'org', 'net', 'edu', 'gov'}
SUSPICIOUS_EXTENSIONS = {'.exe', '.zip', '.bin', '.apk', '.cmd', '.js', '.1exe'}

FEATURE_NAMES = [
    'url_length',
    'has_ip_address',
    'dot_count',
    'https_flag',
    'url_entropy',
    'token_count',
    'subdomain_count',
    'query_param_count',
    'tld_length',
    'path_length',
    'has_hyphen_in_domain',
    'number_of_digits',
    'tld_popularity',
    'suspicious_file_extension',
    'domain_name_length',
    'percentage_numeric_chars'
]

def calculate_shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    length = len(s)
    counts = Counter(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

def extract_features_from_url(url: str) -> Dict[str, Any]:
    """
    Extracts the 16 features from a raw URL string matching LegitPhish dataset definitions.
    Replicates exact behavior on both standard and scheme-less URLs.
    """
    if not isinstance(url, str):
        url = str(url) if url is not None else ""
    url = url.strip()
    
    if not url:
        return {feat: 0.0 if feat != 'query_param_count' else 1 for feat in FEATURE_NAMES}

    url_length = len(url)
    dot_count = url.count('.')
    
    # In original dataset, urlparse is called directly on the raw URL string
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse('')

    https_flag = 1 if url.lower().startswith('https://') else 0

    netloc_raw = parsed.netloc or ""
    
    # Check IP address on stripped host if netloc exists, else check first path token
    has_ip = 0
    if netloc_raw:
        host_only = netloc_raw.split(':')[0].strip('[]')
        try:
            ipaddress.ip_address(host_only)
            has_ip = 1
        except ValueError:
            has_ip = 0
    else:
        # For scheme-less URLs where host ended up in path
        first_token = (parsed.path or "").split('/')[0].split(':')[0].strip('[]')
        try:
            ipaddress.ip_address(first_token)
            has_ip = 1
        except ValueError:
            has_ip = 0

    # Dataset-exact domain splitting: parsed.netloc.split('.') directly
    parts = netloc_raw.split('.')
    if len(parts) >= 2:
        tld = parts[-1].lower()
        domain_name = parts[-2]
        subdomain_count = len(parts) - 2
        tld_length = len(tld)
        domain_name_length = len(domain_name)
    else:
        tld = ""
        domain_name = ""
        subdomain_count = len(parts) - 2  # evaluates to -1 when netloc is empty (''.split('.') -> [''])
        tld_length = 0
        domain_name_length = 0

    has_hyphen_in_domain = 1 if '-' in domain_name else 0

    # Entropy
    url_entropy = calculate_shannon_entropy(url)

    # Token Count: split on [/.\?#=]
    token_count = len(re.split(r'[/.\?#=]', url))

    # Query Param Count: dataset exact len(query.split('&')) where empty query yields 1
    query_str = parsed.query or ""
    query_param_count = len(query_str.split('&'))

    # Path Length
    path_str = parsed.path or ""
    path_length = len(path_str)

    # Digits and Percentage
    num_digits = sum(c.isdigit() for c in url)
    pct_numeric = (num_digits / url_length * 100.0) if url_length > 0 else 0.0

    # TLD Popularity
    tld_clean = tld.split(':')[0]
    tld_popularity = 1 if tld_clean in POPULAR_TLDS else 0

    # Suspicious File Extension
    root, ext = os.path.splitext(path_str)
    ext_lower = ext.lower()
    suspicious_ext = 1 if ext_lower in SUSPICIOUS_EXTENSIONS else 0

    return {
        'url_length': int(url_length),
        'has_ip_address': int(has_ip),
        'dot_count': int(dot_count),
        'https_flag': int(https_flag),
        'url_entropy': float(url_entropy),
        'token_count': int(token_count),
        'subdomain_count': int(subdomain_count),
        'query_param_count': int(query_param_count),
        'tld_length': int(tld_length),
        'path_length': int(path_length),
        'has_hyphen_in_domain': int(has_hyphen_in_domain),
        'number_of_digits': int(num_digits),
        'tld_popularity': int(tld_popularity),
        'suspicious_file_extension': int(suspicious_ext),
        'domain_name_length': int(domain_name_length),
        'percentage_numeric_chars': float(pct_numeric)
    }

def extract_features_vector(url: str) -> List[float]:
    """Returns the ordered list of 16 float features matching FEATURE_NAMES."""
    feats = extract_features_from_url(url)
    return [float(feats[name]) for name in FEATURE_NAMES]
