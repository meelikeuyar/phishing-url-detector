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
    """Tek bir URL'den özellik sözlüğü çıkarır.

    NOT: 'www.' öneki, hesaplamalardan önce normalize ediliyor. Sebebi: www.
    varlığı; url_length, hostname_length, count_dot, subdomain_count gibi
    feature'lara da otomatik olarak sızıyor (çünkü 'www.' eklemek 4 karakter +
    1 nokta + 1 subdomain seviyesi demek). Bu sızıntı yüzünden model, has_www
    feature'ını çıkarsak bile aynı bilgiyi diğer feature'lardan dolaylı olarak
    yeniden inşa edebiliyordu (test: has_www/has_https çıkarıldıktan sonra bile
    google.com / www.google.com skorları arasında ~90 puanlık fark kalıyordu).
    Çözüm: uzunluk/sayma bazlı tüm feature'lar normalize edilmiş (www. hariç)
    hostname/URL üzerinden hesaplanıyor; has_www bilgisi ise TEK ve temiz bir
    flag olarak korunuyor, böylece model ona istismar değil gerçek ağırlığı
    kadar erişebiliyor."""
    url = str(url)
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = parsed.hostname or ""
        path = parsed.path or ""
    except Exception:
        hostname, path = "", ""

    has_www_flag = 1 if hostname.startswith("www.") else 0
    norm_hostname = hostname[4:] if has_www_flag else hostname
    norm_url = url.replace(hostname, norm_hostname, 1) if hostname else url

    hostname_digits = sum(c.isdigit() for c in norm_hostname)
    tld = norm_hostname.rsplit(".", 1)[-1].lower() if "." in norm_hostname else ""

    return {
        "url_length": len(norm_url),
        "hostname_length": len(norm_hostname),
        "path_length": len(path),
        "count_dot": norm_url.count("."),
        "count_hyphen": norm_url.count("-"),
        "count_slash": norm_url.count("/"),
        "count_at": norm_url.count("@"),
        "count_question": norm_url.count("?"),
        "count_equal": norm_url.count("="),
        "count_percent": norm_url.count("%"),
        "count_digit": sum(c.isdigit() for c in norm_url),
        "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        "has_https": 1 if url.startswith("https") else 0,
        "has_www": has_www_flag,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_double_slash": 1 if "//" in url[7:] else 0,
        "subdomain_count": len(norm_hostname.split(".")) - 2 if norm_hostname.count(".") >= 2 else 0,
        "hostname_entropy": round(_shannon_entropy(norm_hostname), 4),
        "digit_ratio_hostname": round(hostname_digits / len(norm_hostname), 4) if norm_hostname else 0.0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
    }


def extract_features_batch(urls) -> pd.DataFrame:
    """Bir URL listesi/Series'inden feature DataFrame'i üretir.
    train.py ve app.py aynı fonksiyonu çağırır."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows)[FEATURE_NAMES]
