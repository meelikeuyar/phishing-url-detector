"""
Ortak feature extraction modülü.

Hem eğitim (train.py / notebook) hem de deployment (app/app.py) tarafında
AYNI kod kullanılsın diye buraya taşındı. Önceki versiyonda bu fonksiyon
notebook'ta ve Streamlit app'inde ayrı ayrı, kopyala-yapıştır olarak
yazılmıştı — bu, ileride biri güncellenip diğeri unutulursa (train/serve
skew) sessiz bir hataya yol açar. Tek kaynak (single source of truth).
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

# Yaygın olarak phishing kampanyalarında kötüye kullanılan / ücretsiz
# verilen TLD'ler. Bu liste kesin bir kanıt değildir, sadece zayıf bir
# sinyaldir (yorumlanabilirlik için ayrı bir feature olarak tutuluyor).
SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "club", "tk", "ml", "ga", "cf",
}

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dot",
    "count_hyphen",
    "count_slash",
    "count_at",
    "count_question",
    "count_equal",
    "count_percent",
    "count_digit",
    "has_ip",
    "has_https",
    "has_www",
    "has_at_symbol",
    "has_double_slash",
    "subdomain_count",
    "hostname_entropy",
    "digit_ratio_hostname",
    "suspicious_tld",
]


def _shannon_entropy(s: str) -> float:
    """Bir string'in Shannon entropy'si. Rastgele/DGA-benzeri (algoritmik
    üretilmiş) domainler yüksek entropy gösterme eğilimindedir; bu klasik
    lexical feature setinde eksikti."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(url: str) -> dict:
    """Tek bir URL'den özellik sözlüğü çıkarır."""
    url = str(url)
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = parsed.hostname or ""
        path = parsed.path or ""
    except Exception:
        hostname, path = "", ""

    hostname_digits = sum(c.isdigit() for c in hostname)
    tld = hostname.rsplit(".", 1)[-1].lower() if "." in hostname else ""

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "count_dot": url.count("."),
        "count_hyphen": url.count("-"),
        "count_slash": url.count("/"),
        "count_at": url.count("@"),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_percent": url.count("%"),
        "count_digit": sum(c.isdigit() for c in url),
        "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        "has_https": 1 if url.startswith("https") else 0,
        "has_www": 1 if "www." in url else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_double_slash": 1 if "//" in url[7:] else 0,
        "subdomain_count": len(hostname.split(".")) - 2 if hostname.count(".") >= 2 else 0,
        "hostname_entropy": round(_shannon_entropy(hostname), 4),
        "digit_ratio_hostname": round(hostname_digits / len(hostname), 4) if hostname else 0.0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
    }


def extract_features_batch(urls) -> pd.DataFrame:
    """Bir URL listesi/Series'inden feature DataFrame'i üretir.
    train.py ve app.py aynı fonksiyonu çağırır."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows)[FEATURE_NAMES]
