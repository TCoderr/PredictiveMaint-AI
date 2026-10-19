"""Generate docs/VERI_KAYNAGI.md (from the data + metrics.json) and reports/TEST_REPORT.md (from pytest JUnit XML).
Run: .venv/Scripts/python scripts/build_docs.py
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / 'reports' / 'metrics.json').read_text(encoding='utf-8'))
D = R['data']
zsha = hashlib.sha256((ROOT / 'data' / 'raw' / 'ai4i2020.zip').read_bytes()).hexdigest()

src = f"""# PredictiveMaint AI — Veri Kaynağı ve Tekrarlanabilir İndirme

- **Veri seti:** AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository
- **Sayfa:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **DOI:** 10.24432/C5HS5C · **Lisans:** CC BY 4.0
- **Tanıtıcı makale:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020 (International Conference on Artificial Intelligence for Industries)
- **İndirme adresi:** https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip (indirme tarihi: 19.10.2026)
- **Yerel dosya:** `data/raw/ai4i2020.zip` (içinde `ai4i2020.csv`) · SHA-256: `{zsha}`

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
| Gözlem / sütun | {D['rows']} / {D['cols']} |
| Eksik değer / tekrarlanan satır | {D['missing']} / {D['duplicates']} |
| Hedef `Machine failure` = 1 | {D['failures']} (%{100 * D['failure_rate']:.2f}) |
| Arıza türleri (TWF, HDF, PWF, OSF, RNF) | {', '.join(f"{k} {v}" for k, v in D['mode_counts'].items())} |
| Arıza = 1 ama hiçbir tür yok | {D['failure_without_mode']} |
| Tür işaretli ama arıza = 0 | {D['mode_without_failure']} (hepsi: {', '.join(f"{k} {v}" for k, v in D['mode_without_failure_by_mode'].items() if v)}) |
| Ürün tipi dağılımı | {', '.join(f"{k} {v}" for k, v in D['type_counts'].items())} |

**UCI sayfasıyla farklar:** Sayfa TWF için 120, RNF için 5 vaka bildiriyor. Dosyada TWF {D['mode_counts']['TWF']}, RNF {D['mode_counts']['RNF']}. Bu projede dosyadaki gerçek değerler kullanıldı.

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
"""
(ROOT / 'docs' / 'VERI_KAYNAGI.md').write_text(src, encoding='utf-8')

root = ET.parse(ROOT / 'reports' / 'test_results.xml').getroot()
s = root if root.tag == 'testsuite' else root.find('testsuite')
tot, fail, err = int(s.get('tests')), int(s.get('failures')), int(s.get('errors'))
lines = ['# PredictiveMaint AI — Test ve Yeniden Üretilebilirlik Raporu', '',
         f'**Tarih:** {s.get("timestamp", "")[:19].replace("T", " ")} · **Ortam:** Python {R["versions"]["python"]}, scikit-learn {R["versions"]["sklearn"]}',
         f'**Sonuç:** {tot} test · {tot - fail - err} başarılı · {fail} başarısız · {err} hata', '',
         '| # | Test | Sonuç | Süre (sn) |', '|---|---|---|---|']
for k, tc in enumerate(s.iter('testcase'), 1):
    res = 'BAŞARISIZ' if tc.find('failure') is not None else 'HATA' if tc.find('error') is not None else 'BAŞARILI'
    lines.append(f'| {k} | `{tc.get("name")}` | {res} | {float(tc.get("time", 0)):.3f} |')
lines += ['', '## Ne doğrulanıyor?', '',
          '- **Veri:** Raporlanan satır, arıza ve arıza türü sayıları ile etiket tutarsızlıkları veriden yeniden sayılıyor.',
          '- **Sızıntı:**',
          '  - Hiçbir modelde UDI, Product ID, Machine failure veya TWF…RNF yok.',
          '  - Bölme katmanlı ve kümeler ayrık (6000/2000/2000).',
          '  - Ölçekleyici yalnızca eğitim verisine fit ediliyor.',
          '- **Yeniden üretim:**',
          '  - LogReg (temel) ve Random Forest, notebook\'tan bağımsız olarak yeniden eğitiliyor.',
          '  - Doğrulama PR-AUC, test metrikleri, karışıklık matrisi ve doğrulama verisiyle seçilen eşik `reports/metrics.json` ile birebir karşılaştırılıyor.',
          '- **Seçim kuralları:**',
          '  - Final model doğrulama PR-AUC değerine göre seçiliyor.',
          '  - FNR = 1 − recall.',
          '  - Naif modelin accuracy değeri yüksek ama recall değeri 0.',
          '- **Notebook:** Bütün hücreler çalıştırılmış ve hata yok; gerekli görseller mevcut.',
          '', '## Notebook\'u yeniden çalıştırma', '',
          '```powershell', '.venv\\Scripts\\python scripts\\run_notebook.py', '```',
          'Notebook bu ortamda iki kez baştan çalıştırıldı ve aynı sayılar üretildi. Bütün rastgelelik `random_state=42` ile sabitlendi.']
(ROOT / 'reports' / 'TEST_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('docs/VERI_KAYNAGI.md, reports/TEST_REPORT.md')
