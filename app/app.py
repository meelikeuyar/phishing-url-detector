import json
import sys
from datetime import datetime, timezone
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

st.set_page_config(page_title="URL Risk Assessment", page_icon="◆", layout="wide")

# ----------------------------------------------------------------------
# TASARIM SİSTEMİ — kurumsal denetim raporu estetiği
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #F5F6F8;
        --surface: #FFFFFF;
        --border: #E2E5EA;
        --text: #10162B;
        --text-dim: #5B6472;
        --brand: #1F3A93;
        --brand-tint: #EEF1FA;
        --safe: #17643A;
        --safe-tint: #EAF6EE;
        --risk: #B42318;
        --risk-tint: #FBEAE8;
        --warn: #B54708;
        --warn-tint: #FCF3E7;
        --serif: 'IBM Plex Serif', Georgia, serif;
        --sans: 'Inter', sans-serif;
        --mono: 'IBM Plex Mono', monospace;
    }

    .stApp { background: var(--bg); font-family: var(--sans); color: var(--text); }

    /* Streamlit'in üst dekor barını ve olası koyu-tema kalıntılarını sıfırla */
    header[data-testid="stHeader"] { background: var(--bg) !important; }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: var(--text); }

    /* Sidebar: başlık, metin, caption hepsi okunur kontrastta olsun */
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: var(--text-dim) !important; }

    /* Slider rengi marka lacivertine bağlansın (varsayılan kırmızı yerine) */
    div[data-testid="stSlider"] [role="slider"] { background-color: var(--brand) !important; border-color: var(--brand) !important; }
    div[data-testid="stSlider"] div[style*="background-color: rgb(255"] { background: var(--brand) !important; }

    /* Aktif sekme alt çizgisi marka lacivertinde olsun */
    .stTabs [aria-selected="true"] { color: var(--brand) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--brand) !important; }
    .stTabs [data-baseweb="tab-border"] { background-color: var(--border) !important; }

    /* Rapor başlığı / letterhead */
    .ug-letterhead {
        display: flex; justify-content: space-between; align-items: flex-end;
        padding-bottom: 18px; margin-bottom: 24px;
        border-bottom: 2px solid var(--text);
    }
    .ug-mark { display: flex; align-items: center; gap: 12px; }
    .ug-mark-glyph {
        width: 34px; height: 34px; border: 2px solid var(--text);
        display: flex; align-items: center; justify-content: center;
        font-family: var(--serif); font-size: 17px; font-weight: 600;
    }
    .ug-title { font-family: var(--serif); font-size: 23px; font-weight: 600; margin: 0; letter-spacing: -0.2px; }
    .ug-subtitle { color: var(--text-dim); font-size: 13px; margin-top: 1px; }
    .ug-meta { text-align: right; font-family: var(--mono); font-size: 11px; color: var(--text-dim); line-height: 1.6; }
    .ug-meta b { color: var(--text); }

    /* İstatistik satırı */
    .ug-stats { display: flex; gap: 0; margin-bottom: 28px; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--surface); }
    .ug-stat { flex: 1; padding: 14px 18px; border-right: 1px solid var(--border); }
    .ug-stat:last-child { border-right: none; }
    .ug-stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
    .ug-stat-value { font-family: var(--mono); font-size: 18px; font-weight: 600; color: var(--text); margin-top: 3px; }

    /* Kartlar */
    .ug-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 24px; margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(16,22,43,0.04);
    }
    .ug-card h4 {
        margin-top: 0; font-size: 12px; color: var(--text-dim); font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.6px; border-bottom: 1px solid var(--border);
        padding-bottom: 10px; margin-bottom: 14px;
    }

    /* Karar rozeti */
    .ug-badge {
        display: inline-block; font-family: var(--mono); font-size: 13px; font-weight: 600;
        letter-spacing: 0.4px; padding: 5px 12px; border-radius: 4px; text-transform: uppercase;
    }
    .ug-badge-safe { background: var(--safe-tint); color: var(--safe); border: 1px solid var(--safe); }
    .ug-badge-warn { background: var(--warn-tint); color: var(--warn); border: 1px solid var(--warn); }
    .ug-badge-risk { background: var(--risk-tint); color: var(--risk); border: 1px solid var(--risk); }

    .ug-url-line { font-family: var(--mono); font-size: 13px; color: var(--text-dim); word-break: break-all; margin-top: 10px; }

    /* Streamlit yerel bileşenleri */
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background: var(--surface) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
        font-family: var(--mono) !important; font-size: 13px !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--surface) !important; border: 1px dashed var(--border) !important; border-radius: 6px !important;
    }
    button[kind="primary"] {
        background: var(--brand) !important; color: #FFFFFF !important;
        border: none !important; font-weight: 500 !important; border-radius: 6px !important;
    }
    button[kind="secondary"] {
        background: var(--surface) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 6px !important;
    }
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 6px; padding: 12px 16px;
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { font-family: var(--sans); font-weight: 500; font-size: 13px; }
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
        return "risk", "MALICIOUS"
    if prob >= threshold * 0.5:
        return "warn", "SUSPICIOUS"
    return "safe", "BENIGN"


def badge_html(band, label):
    return f'<span class="ug-badge ug-badge-{band}">{label}</span>'


def linear_meter_svg(prob: float, band: str) -> str:
    """Risk skorunu, denetim raporlarındaki gibi düz bir gösterge çubuğu olarak çizer."""
    color = {"safe": "#17643A", "warn": "#B54708", "risk": "#B42318"}[band]
    width, height, track_h, pad = 560, 54, 8, 30
    track_w = width - 2 * pad
    marker_x = pad + prob * track_w

    def anchor_and_x(x, text_w_estimate=34):
        if x < pad + text_w_estimate / 2:
            return "start", pad
        if x > width - pad - text_w_estimate / 2:
            return "end", width - pad
        return "middle", x

    ticks = "".join(
        f'<line x1="{pad + t*track_w}" y1="{height/2 - track_h/2 - 5}" '
        f'x2="{pad + t*track_w}" y2="{height/2 + track_h/2 + 5}" '
        f'stroke="#C7CCD6" stroke-width="1"/>'
        for t in (0, 0.25, 0.5, 0.75, 1.0)
    )
    tick_labels = "".join(
        f'<text x="{x}" y="{height/2 + track_h/2 + 18}" text-anchor="{anc}" '
        f'font-family="IBM Plex Mono, monospace" font-size="10" fill="#8A93A3">{int(t*100)}</text>'
        for t, (anc, x) in ((t, anchor_and_x(pad + t*track_w, 20)) for t in (0, 0.25, 0.5, 0.75, 1.0))
    )
    label_anchor, label_x = anchor_and_x(marker_x, 46)
    return f"""
    <svg width="{width}" height="{height+14}" viewBox="0 0 {width} {height+14}">
        <rect x="{pad}" y="{height/2 - track_h/2}" width="{track_w}" height="{track_h}" rx="4" fill="#EDEFF3"/>
        <rect x="{pad}" y="{height/2 - track_h/2}" width="{max(prob*track_w, 3)}" height="{track_h}" rx="4" fill="{color}"/>
        {ticks}
        {tick_labels}
        <polygon points="{marker_x-6},{height/2 - track_h/2 - 12} {marker_x+6},{height/2 - track_h/2 - 12} {marker_x},{height/2 - track_h/2 - 3}"
                 fill="{color}"/>
        <text x="{label_x}" y="{height/2 - track_h/2 - 16}" text-anchor="{label_anchor}"
              font-family="IBM Plex Mono, monospace" font-size="14" font-weight="600" fill="{color}">{prob:.1%}</text>
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
    "Örnek: güvenli": "https://www.wikipedia.org/wiki/Phishing",
    "Örnek: şüpheli": "http://paypal-secure-login.account-verify.xyz/update?user=admin",
    "Örnek: IP tabanlı": "http://192.168.1.1/login/verify?user=admin&pass=1234",
}

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------------
# LETTERHEAD (rapor başlığı)
# ----------------------------------------------------------------------
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.markdown(
    f"""
    <div class="ug-letterhead">
        <div class="ug-mark">
            <div class="ug-mark-glyph">UG</div>
            <div>
                <p class="ug-title">URL Risk Assessment</p>
                <p class="ug-subtitle">Lexical analiz tabanlı phishing / malicious URL tespit sistemi</p>
            </div>
        </div>
        <div class="ug-meta">
            <div>Model: <b>XGBoost (tuned)</b></div>
            <div>Oturum: <b>{now}</b></div>
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
    cm = te.get("confusion_matrix", {})
    f1 = None
    if cm:
        tp, fp, fn = cm.get("tp", 0), cm.get("fp", 0), cm.get("fn", 0)
        if (2 * tp + fp + fn) > 0:
            f1 = 2 * tp / (2 * tp + fp + fn)
    stats = [
        ("Test ROC-AUC", f"{te.get('roc_auc', 0):.4f}" if te.get("roc_auc") else "—"),
        ("F1-Score", f"{f1:.4f}" if f1 else "—"),
        ("Eğitim verisi", "450K+ URL"),
        ("Leakage kontrolü", f"Δ{dl.get('difference', 0):.4f}" if dl else "—"),
    ]
    stats_html = "".join(
        f'<div class="ug-stat"><div class="ug-stat-label">{label}</div>'
        f'<div class="ug-stat-value">{value}</div></div>'
        for label, value in stats
    )
    st.markdown(f'<div class="ug-stats">{stats_html}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Analiz Ayarları**")
    threshold = st.slider(
        "Karar eşiği", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Bu değerin üzerindeki risk skorları 'malicious' sayılır. Düşürürsen "
             "model daha fazla URL'yi şüpheli işaretler (recall artar, precision düşer).",
    )
    st.caption(f"Eşik %{threshold*100:.0f} — bu değerin üzeri MALICIOUS olarak sınıflandırılır.")
    st.divider()
    st.caption(
        "Model yalnızca URL string'ine bakar (lexical özellikler). WHOIS, SSL "
        "sertifikası ya da sayfa içeriği kullanılmaz — tek başına kesin bir güvenlik "
        "kararı için yeterli değildir."
    )

# ----------------------------------------------------------------------
# SEKMELER
# ----------------------------------------------------------------------
tab_single, tab_csv, tab_bulk, tab_model = st.tabs(
    ["Tek URL", "CSV Yükle", "Toplu Giriş", "Model Raporu"]
)

# --- TAB 1: TEK URL ---
with tab_single:
    st.markdown("###### Hızlı örnekler")
    ex_cols = st.columns(len(EXAMPLE_URLS))
    picked = None
    for col, (label, url) in zip(ex_cols, EXAMPLE_URLS.items()):
        if col.button(label, use_container_width=True):
            picked = url

    url_input = st.text_input(
        "URL girin",
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
                band, label = risk_band(prob, threshold)
                st.session_state.history.insert(0, {"URL": target_url, "Risk": f"{prob:.1%}", "Sonuç": label})
                st.session_state.history = st.session_state.history[:8]

                st.markdown('<div class="ug-card"><h4>Değerlendirme Sonucu</h4>', unsafe_allow_html=True)
                st.markdown(badge_html(band, label), unsafe_allow_html=True)
                st.markdown(f'<div class="ug-url-line">{target_url}</div>', unsafe_allow_html=True)
                st.markdown(linear_meter_svg(prob, band), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("Feature bazında detaylı analiz"):
                    feats = extract_features_batch([target_url]).T
                    feats.columns = ["Değer"]
                    feats.index = [FEATURE_LABELS.get(i, i) for i in feats.index]
                    st.dataframe(feats, use_container_width=True)

    if st.session_state.history:
        st.markdown("###### Son analizler (bu oturum)")
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
                st.error("CSV dosyanda 'url' adında bir sütun bulunamadı.")
            else:
                st.success(f"{len(input_df)} URL yüklendi.")
                st.dataframe(input_df.head(), use_container_width=True)

                if st.button("Analiz Başlat", type="primary"):
                    preds, probs = predict(input_df["url"].astype(str), model, feature_names)
                    labels = [risk_band(p, threshold)[1] for p in probs]
                    result_df = pd.DataFrame({
                        "URL": input_df["url"], "Sonuç": labels, "Risk": [f"{p:.1%}" for p in probs],
                    })

                    mal_count = sum(1 for p in probs if p >= threshold)
                    ben_count = len(probs) - mal_count
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Toplam URL", len(result_df))
                    c2.metric("Malicious", mal_count)
                    c3.metric("Benign", ben_count)

                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Sonuçları CSV Olarak İndir", data=csv_data,
                        file_name="phishing_analiz_sonuclari.csv", mime="text/csv",
                    )

# --- TAB 3: TOPLU MANUEL GİRİŞ ---
with tab_bulk:
    bulk_input = st.text_area(
        "Birden fazla URL girin (her satıra bir URL)",
        placeholder="https://www.google.com\nhttp://suspicious-site.com/login",
        height=160,
    )

    if st.button("Toplu Analiz Et", type="primary"):
        if not bulk_input.strip():
            st.warning("Lütfen en az bir URL girin.")
        else:
            urls = [u.strip() for u in bulk_input.strip().split("\n") if u.strip()]
            preds, probs = predict(urls, model, feature_names)
            labels = [risk_band(p, threshold)[1] for p in probs]
            result_df = pd.DataFrame({
                "URL": urls, "Sonuç": labels, "Risk": [f"{p:.1%}" for p in probs],
            })
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "CSV Olarak İndir", data=csv_data,
                file_name="phishing_analiz_sonuclari.csv", mime="text/csv",
            )

# --- TAB 4: MODEL RAPORU ---
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
                st.write(f"Fark: **{dl.get('difference'):.4f}** — anlamsız seviyede, leakage yok.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ug-card"><h4>Final Model — Test Seti</h4>', unsafe_allow_html=True)
            if te:
                cm = te.get("confusion_matrix", {})
                m1, m2 = st.columns(2)
                m1.metric("ROC-AUC", f"{te.get('roc_auc', 0):.4f}")
                m2.metric("Model", "XGBoost (tuned)")
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