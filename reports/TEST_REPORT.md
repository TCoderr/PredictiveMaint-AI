# PredictiveMaint AI — Test ve Yeniden Üretilebilirlik Raporu

**Tarih:** 2026-10-19 02:53:12 · **Ortam:** Python 3.12.10, scikit-learn 1.9.1
**Sonuç:** 11 test · 11 başarılı · 0 başarısız · 0 hata

| # | Test | Sonuç | Süre (sn) |
|---|---|---|---|
| 1 | `test_dataset_facts` | BAŞARILI | 0.016 |
| 2 | `test_label_inconsistencies_are_reported_correctly` | BAŞARILI | 0.002 |
| 3 | `test_no_leaky_columns_in_any_model` | BAŞARILI | 0.000 |
| 4 | `test_stratified_disjoint_split` | BAŞARILI | 0.032 |
| 5 | `test_scaler_is_fit_on_training_data_only` | BAŞARILI | 0.008 |
| 6 | `test_logreg_baseline_reproduces` | BAŞARILI | 0.023 |
| 7 | `test_random_forest_and_threshold_reproduce` | BAŞARILI | 0.599 |
| 8 | `test_selected_model_is_best_on_validation_not_test` | BAŞARILI | 0.000 |
| 9 | `test_accuracy_paradox_documented` | BAŞARILI | 0.000 |
| 10 | `test_notebook_executed_without_errors` | BAŞARILI | 0.046 |
| 11 | `test_at_least_three_eda_figures` | BAŞARILI | 0.001 |

## Ne doğrulanıyor?

- **Veri:** Raporlanan satır, arıza ve arıza türü sayıları ile etiket tutarsızlıkları veriden yeniden sayılıyor.
- **Sızıntı:**
  - Hiçbir modelde UDI, Product ID, Machine failure veya TWF…RNF yok.
  - Bölme katmanlı ve kümeler ayrık (6000/2000/2000).
  - Ölçekleyici yalnızca eğitim verisine fit ediliyor.
- **Yeniden üretim:**
  - LogReg (temel) ve Random Forest, notebook'tan bağımsız olarak yeniden eğitiliyor.
  - Doğrulama PR-AUC, test metrikleri, karışıklık matrisi ve doğrulama verisiyle seçilen eşik `reports/metrics.json` ile birebir karşılaştırılıyor.
- **Seçim kuralları:**
  - Final model doğrulama PR-AUC değerine göre seçiliyor.
  - FNR = 1 − recall.
  - Naif modelin accuracy değeri yüksek ama recall değeri 0.
- **Notebook:** Bütün hücreler çalıştırılmış ve hata yok; gerekli görseller mevcut.

## Notebook'u yeniden çalıştırma

```powershell
.venv\Scripts\python scripts\run_notebook.py
```
Notebook bu ortamda iki kez baştan çalıştırıldı ve aynı sayılar üretildi. Bütün rastgelelik `random_state=42` ile sabitlendi.
