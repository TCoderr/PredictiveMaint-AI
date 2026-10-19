# PredictiveMaint AI — Final Raporu

**Author / Maintainer:** Tugay Sinan Ergün · **GitHub:** TCoderr · **Tarih:** 19 Ekim 2026
**Eğitim:** Makine ve Derin Öğrenme Uygulamaları

## 1. Sonuç

Çalışma tamamlandı ve doğrulandı:
- **Notebook:** Resmî UCI verisiyle 19 kod hücresi hatasız çalıştı. Yeniden çalıştırıldığında aynı sonuçları üretiyor.
- **Testler:** 11 testin tamamı geçti. Testler en önemli sonuçları notebook'tan bağımsız olarak yeniden hesaplıyor.
- **Rapor:** 4 sayfalık rapor `metrics.json`'dan üretildi; içindeki 20 sayı otomatik karşılaştırmayla doğrulandı.

**Kod bağlantısı:** https://github.com/TCoderr/PredictiveMaint-AI (public). Proje sahibinin onayıyla oluşturuldu ve doğrulandı. Rapora GitHub ve Colab açma bağlantısı eklendi.

**Açık konu:** Notebook Colab'da çalıştırılmadı.

## 2. Temel bulgular

- **Veri:** 10.000 sentetik gözlem, 339 arıza (%3,39). Dosyada UCI sayfasından farklı TWF ve RNF sayıları ve 27 tutarsız etiket var: 9 arızanın türü yok; 18 RNF işaretli satırda `Machine failure` = 0. Bunlar raporda açıklandı.
- **Model seçimi:** Doğrulama PR-AUC değerleri:

  | Model | Doğrulama PR-AUC |
  |---|---|
  | **Random Forest (seçilen)** | **0,8668** |
  | MLP | 0,7231 |
  | LogReg (+türetilmiş) | 0,4725 |
  | LogReg (temel) | 0,4054 |

- **Test sonuçları (seçilen model, eşik 0,23):** Eşik yalnızca doğrulama verisiyle seçildi.
  - Recall 0,853, precision 0,879, F1 0,866.
  - 68 gerçek arızadan 10'u kaçırıldı (FNR 0,147).
  - 8 yanlış alarm var.
  - Eşik 0,5'te kaçırılan arıza 15, yanlış alarm 1 idi.
- **Accuracy paradoksu:** "Hep arıza yok" diyen naif model 0,966 accuracy alıyor ama hiçbir arızayı yakalamıyor.
- **Hata analizi:** Kaçırılan 10 arızanın 8'i TWF (test kümesinde 9 TWF arızası var), 2'si türü olmayan arıza. HDF, PWF ve OSF arızalarının hepsi yakalandı.

## 3. Doğrulama

| Kontrol | Sonuç | Kanıt |
|---|---|---|
| Otomatik testler | 11/11 | `reports/TEST_REPORT.md` |
| Notebook yeniden çalıştırma | metrics.json birebir aynı | `reports/reproducibility_check.txt` |
| Rapor ve README sayıları | metrics.json ile aynı | `reports/delivery_verification.txt` |
| Rapor uzunluğu | 4 sayfa (Word) | `reports/office_open_check.json` |
| ZIP | CRC, içerik, `.venv` yok | `reports/zip_verification.txt` |

**Çalışma sırasında bulunup düzeltilen sorunlar:**
- **Grafik hatası:** Karışıklık matrisi çiziminde float/int biçim hatası vardı.
- **Yanlış yorumlar:** Notebook'taki iki sabit yorum gerçek sonuçlarla çelişiyordu: "RNF en çok kaçırılır" (testte RNF arızası yok) ve "bazı yanlış alarmlarda tür işareti var" (yok). Bu yorumlar sonuçlardan otomatik üretilen metinle değiştirildi.
- **Gereksiz uyarı:** joblib temizlik uyarısı, permutation importance tek iş parçacığına alınarak giderildi.
- **Biçim:** Rapordaki ham markdown ters tırnakları temizlendi.

## 4. Kurulan bağımlılıklar

Proje içi `.venv` (Python 3.12.10):
- **Çalışma için:** pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2.
- **Test ve belge üretimi için:** pytest 9.1.1, nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, python-docx 1.2.0, pypdf 6.21.1.

GPU, PyTorch veya ücretli hizmet kullanılmadı.

## 5. Kullanıcı aksiyonları

1. Notebook'u Colab'da bir kez çalıştır (README'deki "Colab'da aç" bağlantısı).

## 6. Araçlar

Kod ve belgeler Claude Code (Anthropic) yardımıyla geliştirildi ve test edildi. Proje sahibi ve sorumlusu Tugay Sinan Ergün'dür.
