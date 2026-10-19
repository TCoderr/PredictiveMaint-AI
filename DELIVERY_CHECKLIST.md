# PredictiveMaint AI — Teslim Kontrol Listesi

**Proje:** PredictiveMaint AI — Endüstriyel Ekipman Arızası Tahmini · **Eğitim:** Makine ve Derin Öğrenme Uygulamaları
**Author / Maintainer:** Tugay Sinan Ergün · **GitHub:** TCoderr · **Tarih:** 19 Ekim 2026
**Yönerge:** `Makine ve Derin Öğrenme Uygulamaları Proje Yönergesi.2026.docx` (yerelde bulundu ve esas alındı)

Durum: ✅ tamamlandı ve doğrulandı · ⚠️ tamamlandı, açık not var · 👤 kullanıcı yapacak

## Yönergenin zorunlu adımları (Bölüm 3) ve rapor şablonu (Bölüm 9)

| # | Gereksinim | Rapordaki yeri / dosya | Kanıt | Durum |
|---|---|---|---|---|
| 3.1 | Problem tanımı (girdi, çıktı, ihtiyaç) | Rapor §1, notebook §1 | — | ✅ |
| 3.2 | Veri: kaynak, örnek sayısı, değişkenler, hedef, dengesizlik | Rapor §2, `docs/VERI_KAYNAGI.md` | 10.000 gözlem; %3,39 arıza; sentetik olduğu belirtildi | ✅ |
| 3.3 | En az 3 anlamlı görsel + yorum | `reports/figures/01–04` (+05–09) | Sınıf dağılımı, özellik dağılımları, arızayla ilişki, korelasyon | ✅ |
| 3.4 | Ön işleme (eksik, aykırı, encoding, scaling, bölme) | Rapor §3, notebook §5 | Pipeline yalnızca eğitime fit (test ile doğrulandı) | ✅ |
| 3.5 | Algoritma ve hiperparametreler | Rapor §4 tablo | 4 model, ızgaralar ve seçilen değerler | ✅ |
| 3.6 | Eğitim / doğrulama yaklaşımı, hiperparametre ayarı | Rapor §4 | Doğrulama PR-AUC ile seçim; test kullanılmadı | ✅ |
| 3.7 | Accuracy, precision, recall, F1, confusion matrix (+grafik) | Rapor §5, Şekil 3 | ROC-AUC ve PR-AUC de var | ✅ |
| 3.8 | Güçlü/zayıf yönler, hatalar, öneriler | Rapor §6–9 | FN bölümü, hata analizi, sınırlar | ✅ |
| 5 | Rapor 3–5 sayfa | `reports/PredictiveMaint_AI_Rapor.pdf` | Word'de 4 sayfa | ✅ |
| 5 | Veri kaynağı bağlantısı | Rapor §2 | UCI DOI | ✅ |
| 5 | Kodun GitHub/Colab bağlantısıyla verilmesi | Rapor başlığı | **Bağlantı henüz yok**; uydurulmadı | 👤 |

## Görev kapsamındaki ek gereksinimler

| Gereksinim | Kanıt | Durum |
|---|---|---|
| Resmî kaynaktan indirme, tekrarlanabilir talimat | `data/raw/ai4i2020.zip` (SHA-256 `docs/VERI_KAYNAGI.md` içinde), notebook otomatik indirme | ✅ |
| Sentetik veri açıklaması | Rapor, README, notebook, veri belgesi | ✅ |
| Kimlik ve sızıntılı sütunların dışlanması (UDI, Product ID, TWF…RNF) | Notebook'ta assert + test | ✅ |
| Katmanlı eğitim/doğrulama/test bölmesi | 6000/2000/2000, test | ✅ |
| En az 2 model (LR + RF), isteğe bağlı MLP | 4 model | ✅ |
| Eşiğin yalnızca doğrulamayla seçilmesi, testte tek değerlendirme | Eşik 0,23; test bağımsız olarak yeniden üretildi | ✅ |
| Accuracy'nin yetersizliği analizi | Naif model karşılaştırması | ✅ |
| FN bölümü (sayı, FNR, recall ilişkisi, önem, P/R dengesi) | Rapor §6, Şekil 4 | ✅ |
| Parasal maliyet varsayımları | Maliyet hesabı **yapılmadı** (gerçek maliyet bilinmiyor) | ✅ |
| LR katsayıları ve RF önemlerinin yorumu, nedensellik uyarısı | Rapor §7, Şekil 5 | ✅ |
| FP/FN örnek analizi | Rapor §8, notebook §10 | ✅ |
| Rapor sayıları = notebook sonuçları (otomatik) | `reports/delivery_verification.txt` (20 değer) | ✅ |
| Notebook yeniden çalıştırılabilir | `reports/reproducibility_check.txt`: aynı metrics.json | ✅ |
| Testler | `reports/TEST_REPORT.md`: 11/11 | ✅ |
| requirements, README | `requirements.txt`, `requirements-dev.txt`, `README.md` | ✅ |
| ZIP | `dist/PredictiveMaint_AI_Teslim.zip` | ✅ |
| Colab'da çalıştırma | Yerelde doğrulandı; **Colab'da çalıştırılmadı** | ⚠️ |

## Kullanıcı aksiyonları

1. 👤 Onay verirsen projeyi GitHub'a (TCoderr) yükle ve bağlantıyı rapora ve README'ye ekle. Rapor `scripts/build_report.py` ile yeniden üretilir.
2. 👤 Notebook'u Google Colab'da bir kez "Tümünü çalıştır" ile doğrula.
