import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features import extract_features_batch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishing_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SHAP_IMG_PATH = ROOT / "reports" / "shap_feature_importance.png"
CM_IMG_PATH = ROOT / "reports" / "confusion_matrix.png"

st.set_page_config(page_title="URL Guard — Phishing Detector", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------
# TASARIM SİSTEMİ
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B1220;
        --surface: #131B2E;
        --surface-2: #182238;
        --border: #223052;
        --text: #E7ECF5;
        --text-dim: #8B93A7;
        --safe: #2DD4BF;
        --risk: #F0475C;
        --warn: #F5A623;
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 15% -10%, rgba(45,212,191,0.07), transparent),
            radial-gradient(900px 500px at 90% 0%, rgba(240,71,92,0.06), transparent),
            var(--bg);
        font-family: var(--sans);
        color: var(--text);
    }

    .ug-hero {
        display: flex; align-items: center; gap: 16px;
        padding: 4px 0 8px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }
    .ug-hero-icon {
        width: 46px; height: 46px; border-radius: 12px;
        background: linear-gradient(135deg, rgba(45,212,191,0.18), rgba(45,212,191,0.04));
        border: 1px solid rgba(45,212,191,0.35);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
    }
    .ug-hero-title { font-family: var(--mono); font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin: 0; }
    .ug-hero-sub { color: var(--text-dim); font-size: 14px; margin-top: 2px; }

    .ug-stats { display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 26px 0; }
    .ug-chip {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 10px 16px;
        display: flex; flex-direction: column; min-width: 120px;
    }
    .ug-chip-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.6px; }
    .ug-chip-value { font-family: var(--mono); font-size: 19px; font-weight: 600; color: var(--safe); margin-top: 2px; }

    .ug-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 14px; padding: 22px; margin-bottom: 16px;
    }
    .ug-card h4 { margin-top: 0; font-size: 15px; color: var(--text-dim); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

    .ug-verdict { display: flex; align-items: center; gap: 18px; padding: 6px 0; }
    .ug-verdict-label { font-family: var(--mono); font-size: 22px; font-weight: 700; }
    .ug-verdict-note { color: var(--text-dim); font-size: 13px; margin-top: 4px; }

    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background: var(--surface-2) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 8px !important;
        font-family: var(--mono) !important; font-size: 14px !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--surface-2) !important; border: 1px dashed var(--border) !important; border-radius: 10px !important;
    }
    button[kind="primary"] {
        background: var(--safe) !important; color: #04201C !important;
        border: none !important; font-weight: 600 !important; border-radius: 8px !important;
    }
    button[kind="secondary"] {
        background: var(--surface-2) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 8px !important;
    }
    div[data-testid="stMetric"] {
        background: var(--surface-2); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 16px;
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { font-family: var(--mono); font-size: 13px; }
    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def predict(urls, model, feature_names):
    feat_df = extract_features_batch(urls)[feature_names]
    preds = model.predict(feat_df)
    probs = model.predict_proba(feat_df)[:, 1]
    return preds, probs


def risk_band(prob, threshold):
    if prob >= threshold:
        return "risk", "🚨", "PHISHING / MALICIOUS"
    if prob >= threshold * 0.5:
        return "warn", "⚠️", "ŞÜPHELİ — dikkatli ol"
    return "safe", "✅", "GÜVENLİ / BENIGN"


def gauge_svg(prob: float, band: str) -> str:
    """Risk skorunu dairesel bir gösterge (gauge) olarak SVG'ye çizer."""
    color = {"safe": "#2DD4BF", "warn": "#F5A623", "risk": "#F0475C"}[band]
    radius, stroke = 64, 12
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - prob)
    size = (radius + stroke) * 2 + 8
    center = size / 2
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                stroke="#1F2A44" stroke-width="{stroke}" />
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                stroke="{color}" stroke-width="{stroke}" stroke-linecap="round"
                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                transform="rotate(-90 {center} {center})"
                style="transition: stroke-dashoffset 0.6s ease;" />
        <text x="{center}" y="{center - 4}" text-anchor="middle"
              font-family="JetBrains Mono, monospace" font-size="26" font-weight="700"
              fill="{color}">{prob:.0%}</text>
        <text x="{center}" y="{center + 18}" text-anchor="middle"
              font-family="Inter, sans-serif" font-size="11" fill="#8B93A7">risk skoru</text>
    </svg>
    """


FEATURE_LABELS = {
    "url_length": "URL uzunluğu",
    "hostname_length": "Hostname uzunluğu",
    "path_length": "Path uzunluğu",
    "count_dot": "Nokta (.) sayısı",
    "count_hyphen": "Tire (-) sayısı",
    "count_slash": "Slash (/) sayısı",
    "count_at": "@ sayısı",
    "count_question": "? sayısı",
    "count_equal": "= sayısı",
    "count_percent": "% sayısı",
    "count_digit": "Rakam sayısı",
    "has_ip": "IP adresi içeriyor mu",
    "has_https": "HTTPS kullanıyor mu",
    "has_www": "www. içeriyor mu",
    "has_at_symbol": "@ sembolü var mı",
    "has_double_slash": "Şüpheli çift slash var mı",
    "subdomain_count": "Subdomain sayısı",
    "hostname_entropy": "Hostname entropy'si (rastgelelik)",
    "digit_ratio_hostname": "Hostname'deki rakam oranı",
    "suspicious_tld": "Şüpheli TLD (.xyz, .top vb.)",
}

EXAMPLE_URLS = {
    "✅ Güvenli örnek": "https://www.wikipedia.org/wiki/Phishing",
    "🚨 Şüpheli örnek": "http://paypal-secure-login.account-verify.xyz/update?user=admin",
    "🚨 IP tabanlı örnek": "http://192.168.1.1/login/verify?user=admin&pass=1234",
}

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------------
# BAŞLIK + MODEL İSTATİSTİKLERİ
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="ug-hero">
        <div class="ug-hero-icon">🛡️</div>
        <div>
            <p class="ug-hero-title">URL GUARD</p>
            <p class="ug-hero-sub">Lexical analiz ile phishing / malicious URL tespiti</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error(
        "Model dosyaları bulunamadı. Önce `python train.py --data <urldata.csv>` çalıştırıp "
        "`models/phishing_model.pkl` ve `models/feature_names.pkl` dosyalarını oluşturman gerekiyor."
    )
    st.stop()

metrics = load_metrics()
if metrics:
    te = metrics.get("test_evaluation", {})
    dl = metrics.get("domain_leakage_check", {})
    cr = te.get("classification_report", {})
    f1 = cr.get("malicious", {}).get("f1-score")
    chips = [
        ("Test ROC-AUC", f"{te.get('roc_auc', 0):.4f}" if te.get("roc_auc") else "—"),
        ("F1 (malicious)", f"{f1:.4f}" if f1 else "—"),
        ("Eğitim örneği", "450K+ URL"),
        ("Domain leakage", f"Δ {dl.get('difference', 0):.4f}" if dl else "—"),
    ]
    chips_html = "".join(
        f'<div class="ug-chip"><div class="ug-chip-label">{label}</div>'
        f'<div class="ug-chip-value">{value}</div></div>'
        for label, value in chips
    )
    st.markdown(f'<div class="ug-stats">{chips_html}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR — karar eşiği
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    threshold = st.slider(
        "Karar eşiği (threshold)", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Bu değerin üzerindeki risk skorları 'malicious' sayılır. Düşürürsen "
             "model daha fazla URL'yi şüpheli işaretler (recall artar, precision düşer).",
    )
    st.caption(
        f"Eşik **{threshold:.0%}** iken: %{threshold*100:.0f} ve üzeri risk skoru "
        "phishing olarak işaretlenir."
    )
    st.divider()
    st.caption(
        "⚠️ Model yalnızca URL string'ine bakar (lexical özellikler). WHOIS, SSL "
        "sertifikası ya da sayfa içeriği kullanılmaz — tek başına kesin bir güvenlik "
        "kararı için yeterli değildir."
    )

# ----------------------------------------------------------------------
# SEKMELER
# ----------------------------------------------------------------------
tab_single, tab_csv, tab_bulk, tab_model = st.tabs(
    ["🔍 Tek URL", "📂 CSV Yükle", "📋 Toplu Giriş", "📊 Model Hakkında"]
)

# --- TAB 1: TEK URL ---
with tab_single:
    st.markdown("##### Hızlı örnekler")
    ex_cols = st.columns(len(EXAMPLE_URLS))
    picked = None
    for col, (label, url) in zip(ex_cols, EXAMPLE_URLS.items()):
        if col.button(label, use_container_width=True):
            picked = url

    url_input = st.text_input(
        "URL girin:",
        value=picked or "",
        placeholder="https://www.example.com",
        help="Analiz etmek istediğin URL'yi buraya yapıştır",
    )

    if st.button("Analiz Et", type="primary") or picked:
        target_url = (picked or url_input).strip()
        if not target_url:
            st.warning("Lütfen bir URL girin.")
        else:
            try:
                preds, probs = predict([target_url], model, feature_names)
                prob = float(probs[0])
            except Exception as e:
                st.error(f"Analiz sırasında hata oluştu: {e}")
            else:
                band, emoji, label = risk_band(prob, threshold)
                st.session_state.history.insert(0, {"URL": target_url, "Risk": f"{prob:.1%}", "Sonuç": label})
                st.session_state.history = st.session_state.history[:8]

                st.markdown('<div class="ug-card">', unsafe_allow_html=True)
                gcol, vcol = st.columns([1, 2])
                with gcol:
                    st.markdown(gauge_svg(prob, band), unsafe_allow_html=True)
                with vcol:
                    color = {"safe": "#2DD4BF", "warn": "#F5A623", "risk": "#F0475C"}[band]
                    st.markdown(
                        f"""
                        <div class="ug-verdict">
                            <span style="font-size:32px">{emoji}</span>
                            <div>
                                <div class="ug-verdict-label" style="color:{color}">{label}</div>
                                <div class="ug-verdict-note">Eşik: %{threshold*100:.0f} · Skor: %{prob*100:.1f}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(target_url, language=None)
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("🔬 Feature bazında detaylı analiz"):
                    feats = extract_features_batch([target_url]).T
                    feats.columns = ["Değer"]
                    feats.index = [FEATURE_LABELS.get(i, i) for i in feats.index]
                    st.dataframe(feats, use_container_width=True)

    if st.session_state.history:
        st.markdown("##### Son analizler (bu oturum)")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

# --- TAB 2: CSV YÜKLE ---
with tab_csv:
    st.markdown("Analiz edilecek URL'lerin bulunduğu, `url` sütunlu bir CSV dosyası yükle.")
    st.code("url\nhttps://www.google.com\nhttp://suspicious-site.com/login", language="csv")

    uploaded_file = st.file_uploader("CSV dosyası seç", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
        else:
            if "url" not in input_df.columns:
                st.error("❌ CSV dosyanda 'url' adında bir sütun bulunamadı.")
            else:
                st.success(f"✅ {len(input_df)} URL yüklendi.")
                st.dataframe(input_df.head(), use_container_width=True)

                if st.button("Analiz Başlat", type="primary"):
                    preds, probs = predict(input_df["url"].astype(str), model, feature_names)
                    labels = [risk_band(p, threshold)[2] for p in probs]
                    result_df = pd.DataFrame({
                        "URL": input_df["url"],
                        "Sonuç": labels,
                        "Risk": [f"{p:.1%}" for p in probs],
                    })

                    mal_count = sum(1 for p in probs if p >= threshold)
                    ben_count = len(probs) - mal_count
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Toplam URL", len(result_df))
                    c2.metric("🚨 Malicious", mal_count)
                    c3.metric("✅ Benign", ben_count)

                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Sonuçları CSV Olarak İndir", data=csv_data,
                        file_name="phishing_analiz_sonuclari.csv", mime="text/csv",
                    )

# --- TAB 3: TOPLU MANUEL GİRİŞ ---
with tab_bulk:
    bulk_input = st.text_area(
        "Birden fazla URL girin (her satıra bir URL):",
        placeholder="https://www.google.com\nhttp://suspicious-site.com/login",
        height=160,
    )

    if st.button("Toplu Analiz Et", type="primary"):
        if not bulk_input.strip():
            st.warning("Lütfen en az bir URL girin.")
        else:
            urls = [u.strip() for u in bulk_input.strip().split("\n") if u.strip()]
            preds, probs = predict(urls, model, feature_names)
            labels = [risk_band(p, threshold)[2] for p in probs]
            result_df = pd.DataFrame({
                "URL": urls, "Sonuç": labels, "Risk": [f"{p:.1%}" for p in probs],
            })
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ CSV Olarak İndir", data=csv_data,
                file_name="phishing_analiz_sonuclari.csv", mime="text/csv",
            )

# --- TAB 4: MODEL HAKKINDA ---
with tab_model:
    if not metrics:
        st.info("`reports/metrics.json` bulunamadı — `train.py` çalıştırıldığında otomatik oluşur.")
    else:
        cv = metrics.get("cross_validation", {})
        tune = metrics.get("hyperparameter_tuning", {})
        te = metrics.get("test_evaluation", {})
        dl = metrics.get("domain_leakage_check", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ug-card"><h4>Cross-Validation (5-Fold)</h4>', unsafe_allow_html=True)
            if cv:
                cv_df = pd.DataFrame([
                    {"Model": k, "Mean AUC": f"{v['mean_auc']:.4f}", "Std": f"±{v['std_auc']:.4f}"}
                    for k, v in cv.items()
                ])
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="ug-card"><h4>Domain-Level Leakage Kontrolü</h4>', unsafe_allow_html=True)
            if dl:
                st.write(f"Yöntem: **{dl.get('method')}**")
                st.write(f"Benzersiz hostname: **{dl.get('unique_hostnames'):,}**")
                st.write(f"Random split AUC: **{dl.get('random_split_auc'):.4f}**")
                st.write(f"Group split AUC: **{dl.get('group_split_auc'):.4f}**")
                st.write(f"Fark: **{dl.get('difference'):.4f}** — anlamsız seviyede, leakage yok ✅")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ug-card"><h4>Final Model — Test Seti</h4>', unsafe_allow_html=True)
            if te:
                cm = te.get("confusion_matrix", {})
                m1, m2 = st.columns(2)
                m1.metric("ROC-AUC", f"{te.get('roc_auc', 0):.4f}")
                m2.metric("En iyi hyperparametreler", "XGBoost (tuned)")
                st.write(f"TN={cm.get('tn')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TP={cm.get('tp')}")
                if tune.get("best_params"):
                    st.json(tune["best_params"])
            st.markdown("</div>", unsafe_allow_html=True)

        if SHAP_IMG_PATH.exists():
            st.markdown('<div class="ug-card"><h4>SHAP Feature Importance</h4>', unsafe_allow_html=True)
            st.image(str(SHAP_IMG_PATH), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if CM_IMG_PATH.exists():
            st.markdown('<div class="ug-card"><h4>Confusion Matrix</h4>', unsafe_allow_html=True)
            st.image(str(CM_IMG_PATH), use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)
