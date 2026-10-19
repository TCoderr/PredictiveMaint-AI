# PredictiveMaint AI — Veri Kaynağı ve Tekrarlanabilir İndirme

- **Veri seti:** AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository
- **Sayfa:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **DOI:** 10.24432/C5HS5C · **Lisans:** CC BY 4.0
- **Tanıtıcı makale:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020 (International Conference on Artificial Intelligence for Industries)
- **İndirme adresi:** https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip (indirme tarihi: 19.10.2026)
- **Yerel dosya:** `data/raw/ai4i2020.zip` (içinde `ai4i2020.csv`) · SHA-256: `f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252`

## Tekrarlanabilir indirme
```powershell
curl -L -o data/raw/ai4i2020.zip "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
```
Notebook, yerel dosya yoksa (ör. Google Colab) aynı adresten otomatik olarak indirir.

## Verinin niteliği
UCI açıklamasına göre veri seti **sentetiktir**: gerçek kestirimci bakım verisini yansıtmak için üretilmiştir, gerçek bir fabrikadan ölçülmemiştir.

## Dosyada doğrulanan özellikler (notebook çıktısı)
| Özellik | Değer |
|---|---|
| Gözlem / sütun | 10000 / 14 |
| Eksik değer / tekrarlanan satır | 0 / 0 |
| Hedef `Machine failure` = 1 | 339 (%3.39) |
| Arıza türleri (TWF, HDF, PWF, OSF, RNF) | TWF 46, HDF 115, PWF 95, OSF 98, RNF 19 |
| Arıza = 1 ama hiçbir tür yok | 9 |
| Tür işaretli ama arıza = 0 | 18 (hepsi: RNF 18) |
| Ürün tipi dağılımı | L 6000, M 2997, H 1003 |

**UCI sayfasıyla farklar:** Sayfa TWF için 120, RNF için 5 vaka bildiriyor. Dosyada TWF 46, RNF 19. Bu projede dosyadaki gerçek değerler kullanıldı.

## Sütunların kullanımı
| Sütun | Kullanım | Gerekçe |
|---|---|---|
| UDI | Kullanılmadı | Sıra numarası (kimlik) |
| Product ID | Kullanılmadı | Ürün kimliği; ilk harfi Type ile aynı, kalanı seri numarası |
| Type | Girdi (one-hot) | Ürün kalite varyantı L/M/H |
| Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] | Girdi (ölçeklenmiş) | Süreç sırasında bilinen ölçümler |
| temp_diff, power_w, strain | Türetilmiş girdi | Aynı andaki ölçümlerden hesaplanır |
| Machine failure | Hedef | — |
| TWF, HDF, PWF, OSF, RNF | **Kullanılmadı** (yalnızca hata analizi) | Arıza sonrasında bilinen bilgi: hedef sızıntısı |
