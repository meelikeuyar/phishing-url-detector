 Phishing URL Detector
 Canlı demo: [buraya deploy sonrası link gelecek]
Bir URL'nin sadece metinsel (lexical) özelliklerine bakarak phishing/malicious
olup olmadığını tahmin eden bir makine öğrenmesi projesi. Kaggle'daki
Malicious and Benign URLs
veri seti (450K+ URL) üzerinde eğitildi.
>  **Sınırlama:** Model yalnızca URL string'ine bakar; WHOIS, SSL sertifikası,
> DNS kaydı veya sayfa içeriği gibi bilgileri kullanmaz. Bu nedenle tek başına
> production-grade bir güvenlik kararı için yeterli değildir — bkz. "Bilinen
> Sınırlamalar" bölümü.
Proje Yapısı
```
├── src/features.py      # Tek kaynak feature extraction (train + serve ortak kullanır)
├── train.py              # CV + class imbalance + hyperparameter tuning + SHAP
├── app/app.py             # Streamlit deployment arayüzü
├── models/                # Eğitilmiş model (.pkl) — train.py sonrası oluşur
├── reports/                # metrics.json, confusion_matrix.png, shap_*.png
└── requirements.txt
```
Yaklaşım
Feature engineering — 20 lexical özellik: URL/hostname/path uzunluğu,
özel karakter sayıları, IP kullanımı, subdomain sayısı, hostname entropy'si
(DGA/rastgele domain sinyali) ve şüpheli TLD bayrağı.
Model seçimi — Logistic Regression, Random Forest ve XGBoost,
Stratified 5-Fold cross-validation ile karşılaştırıldı (tek train/test
split yerine).
Class imbalance — veri setinde benign:malicious oranı yaklaşık 3.3:1.
`class_weight="balanced"` (LogReg/RF) ve `scale_pos_weight` (XGBoost) ile
ele alındı.
Hyperparameter tuning — XGBoost için `RandomizedSearchCV` (ROC-AUC
optimizasyonu).
Explainability — SHAP ile global feature importance ve örnek bazlı
açıklama (`reports/shap_*.png`).
Değerlendirme — Test seti üzerinde ROC-AUC, precision/recall/F1 ve
confusion matrix; false negative (kaçırılan phishing) ile false positive
(yanlış bloklanan güvenli site) arasındaki maliyet farkı ayrıca yorumlandı.
Sonuçlar
Kaggle'daki tam veri seti (450K+ URL, benign:malicious ≈ 3.3:1) üzerinde eğitildi.
Model	CV ROC-AUC (5-fold, mean ± std)
Logistic Regression	0.9979 ± 0.0002
Random Forest	0.9984 ± 0.0002
XGBoost (tuned)	0.9989 ± 0.0001
En iyi hyperparametreler (RandomizedSearchCV, `n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.7, min_child_weight=5`)
Final model — test seti (90.036 örnek):
Metrik	Değer
ROC-AUC	0.9992
Precision (malicious)	0.9946
Recall (malicious)	0.9938
F1-score	0.9942
False Positive Rate	%0.16 (yanlışlıkla bloklanan güvenli site)
False Negative Rate	%0.62 (kaçırılan phishing)
Confusion matrix: TN=69.035, FP=113, FN=129, TP=20.759 — false negative oranının
false positive'ten biraz daha yüksek olması dikkat çekici; phishing tespitinde
FN daha maliyetli olduğundan (kullanıcı zarar görür), production'a alınırsa
karar eşiği (threshold) recall'u artıracak şekilde aşağı çekilebilir.
Overfitting / Domain-Level Leakage Kontrolü
Rastgele URL bazlı split, aynı domain'in bir kısmının train'de bir kısmının test'te
kalmasına izin verir — bu, modelin "phishing kalıbını" değil "bu domaini daha önce
gördüm"ü öğrenmiş olma riskini taşır ve skoru olduğundan iyi gösterebilir (optimistic
bias). Bunu test etmek için, aynı hostname'e sahip tüm URL'lerin zorunlu olarak aynı
fold'da kaldığı StratifiedGroupKFold ile skor tekrar hesaplandı:
Split yöntemi	ROC-AUC
Rastgele (URL bazlı) split	0.9989
Domain-aware (GroupKFold) split	0.9984 ± 0.0002
Fark	0.0005
166.989 benzersiz hostname üzerinden yapılan bu testte fark ölçüm gürültüsü
seviyesinde kaldı. Sonuç: domain leakage yok, model gerçekten genelleşebilen
lexical sinyaller öğrenmiş — sadece daha önce gördüğü domainleri ezberlemiyor.
![Confusion Matrix](reports/confusion_matrix.png)
![SHAP Feature Importance](reports/shap_feature_importance.png)
Kurulum & Çalıştırma
```bash
pip install -r requirements.txt

# 1. Eğitim (urldata.csv Kaggle'dan indirilmiş olmalı)
python train.py --data path/to/urldata.csv

# 2. Streamlit uygulaması
streamlit run app/app.py
```
Bilinen Sınırlamalar
Sadece lexical özellikler: WHOIS/domain yaşı, SSL sertifika bilgisi,
sayfa içeriği gibi daha güçlü sinyaller kullanılmıyor. Saldırgan, modelin
baktığı yüzeysel kalıpları (uzunluk, özel karakter sayısı vb.) taklit ederek
atlatabilir (adversarial evasion).
Statik veri seti: Phishing kalıpları zamanla değişir; model periyodik
olarak yeniden eğitilmeden performansı düşebilir (concept drift). Production
ortamı için bir retraining/monitoring pipeline'ı gerekir.
Tek dil/karakter seti odaklı sezgiler: `has_www`, `has_https` gibi
bazı feature'lar Latin alfabesi ve İngilizce URL kalıplarına göre tasarlandı;
homograph/IDN saldırıları ayrıca ele alınmadı.
Yol Haritası
[ ] WHOIS / domain yaşı feature'ı ekle (network erişimi gerektirir)
[ ] SSL sertifika bilgisi feature'ı ekle
[ ] FastAPI ile hafif bir REST endpoint ekle (Streamlit'e ek olarak)
[ ] Basit unit testler (`src/features.py` için) ve CI (GitHub Actions)
[ ] Model retraining/monitoring için basit bir zamanlanmış pipeline