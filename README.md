# PredictiveMaint AI — Endüstriyel Ekipman Arızası Tahmini

**Eğitim:** Makine ve Derin Öğrenme Uygulamaları — "Makine Öğrenmesi ile Probleme Uygun Çözüm Geliştirme"
**Author / Maintainer:** Tugay Sinan Ergün · **GitHub:** TCoderr

Süreç ölçümlerinden (sıcaklıklar, devir, tork, takım aşınması, ürün tipi) **arıza olasılığını** tahmin eden bir sınıflandırma çalışması:
- Dört model karşılaştırıldı: Logistic Regression (iki varyant), Random Forest ve MLP.
- Değerlendirme dengesiz veriye uygun metriklerle yapıldı.
- Yanlış negatifler (kaçırılan arızalar) ayrıca analiz edildi.
- Model açıklanabilirliği ve hata analizi incelendi.

> **Uyarı:** AI4I 2020 veri seti **sentetiktir**; gerçek bir fabrikadan ölçülmemiştir. Bu çalışma akademik bir prototiptir; modelin gerçek fabrikalarda doğrudan güvenilir şekilde kullanılabileceği iddia edilmez.

## Sonuçlar (test kümesi: 2000 gözlem, 68 arıza)

| Model | Eşik | Precision | Recall | F1 | PR-AUC | Kaçırılan arıza (FN) |
|---|---|---|---|---|---|---|
| LogReg (temel) | 0,5 | 0,692 | 0,132 | 0,222 | 0,416 | 59 |
| **Random Forest (seçilen)** | 0,5 | 0,981 | 0,779 | 0,869 | 0,891 | 15 |
| **Random Forest (seçilen)** | **0,23** (doğrulamada F2 maks.) | **0,879** | **0,853** | **0,866** | 0,891 | **10** |

Bütün modellerin tablosu, karışıklık matrisleri ve hata analizi:
- `reports/PredictiveMaint_AI_Rapor.pdf` (4 sayfa)
- `notebooks/PredictiveMaint_AI.ipynb`
- Ham sonuçlar: `reports/metrics.json`

## Yöntem özeti

- **Veri:** 10.000 gözlem; arıza oranı %3,39. Arıza türü sütunları (TWF…RNF), `UDI` ve `Product ID` hedef sızıntısını önlemek için modele **alınmadı**.
- **Ön işleme:** Katmanlı %60/%20/%20 bölme. Pipeline (StandardScaler + OneHotEncoder) yalnızca eğitim verisine fit edildi. Türetilmiş özellikler: `temp_diff`, `power_w`, `strain`.
- **Model ve eşik seçimi:** Hiperparametreler ve final model **doğrulama PR-AUC** ile seçildi. Karar eşiği **doğrulama** verisinde F2 ile belirlendi. Test kümesi en sonda bir kez kullanıldı.

## Klasör yapısı

```
PredictiveMaint-AI/
├── notebooks/PredictiveMaint_AI.ipynb   # çalıştırılmış, Colab uyumlu notebook (bütün analiz)
├── data/raw/ai4i2020.zip                # UCI resmî arşivi (ai4i2020.csv)
├── docs/VERI_KAYNAGI.md                 # kaynak, lisans, indirme talimatı, sütun kullanımı
├── reports/
│   ├── PredictiveMaint_AI_Rapor.pdf     # 3–5 sayfalık rapor (+ .docx)
│   ├── metrics.json                     # notebook'un ürettiği bütün sayılar
│   ├── figures/                         # 9 görsel (EDA, karışıklık matrisleri, PR eğrileri, eşik, önem)
│   ├── TEST_REPORT.md, test_results.xml # 11 otomatik test
│   └── reproducibility_check.txt        # notebook yeniden çalıştırma karşılaştırması
├── tests/test_predictive_maint.py       # bağımsız yeniden hesaplama testleri
└── scripts/                             # notebook üretimi/çalıştırma, rapor, doğrulama, paketleme
```

## Kurulum ve çalıştırma (Windows, Python 3.12)

```powershell
cd PredictiveMaint-AI
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\run_notebook.py            # notebook'u baştan çalıştırır (~25 sn)
.venv\Scripts\python -m pytest -v                        # 11 test
```

Rapor üretimi için şu komutlar kullanılır (PDF için Microsoft Office gerekir):

```powershell
.venv\Scripts\python scripts\build_report.py
powershell -ExecutionPolicy Bypass -File scripts\export_office.ps1
```

## Google Colab

1. https://colab.research.google.com → **Dosya → Not defteri yükle** → `notebooks/PredictiveMaint_AI.ipynb`.
2. **Tümünü çalıştır.** Notebook, yerel veri yoksa resmî UCI adresinden indirir ve çıktıları `outputs/` klasörüne yazar.

> Notebook yerelde çalıştırıldı ve tekrarlanabilirliği doğrulandı. **Google Colab'da çalıştırılmadı.**

## Kod bağlantısı

Yönerge, kodun raporda GitHub veya Colab bağlantısıyla verilmesini istiyor. **Bağlantı henüz yayımlanmadı.** Proje sahibinin onayıyla GitHub'a (TCoderr) yüklendikten sonra rapordaki ve bu README'deki yer güncellenmelidir.

## Kaynaklar ve lisans

- **Veri:** AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C, **CC BY 4.0**.
- **Makale:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020.
- **Geliştirme:** Kod ve belgeler Claude Code (Anthropic) yardımıyla geliştirildi ve test edildi. Proje sahibi ve sorumlusu Tugay Sinan Ergün'dür.
