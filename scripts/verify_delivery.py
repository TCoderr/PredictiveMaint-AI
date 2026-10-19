"""Verify the PredictiveMaint AI delivery. Writes reports/delivery_verification.txt (with --zip: reports/zip_verification.txt).
Run: .venv/Scripts/python scripts/verify_delivery.py [--zip]
"""
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import nbformat
from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
AUTHOR = 'Tugay Sinan Ergün'
ZIP = ROOT / 'dist' / 'PredictiveMaint_AI_Teslim.zip'
R = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))
results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


def n(x, d=3): return f'{x:.{d}f}'.replace('.', ',')


REQUIRED = ['notebooks/PredictiveMaint_AI.ipynb', 'data/raw/ai4i2020.zip', 'requirements.txt', 'requirements-dev.txt', 'README.md',
            'reports/PredictiveMaint_AI_Rapor.pdf', 'reports/PredictiveMaint_AI_Rapor.docx', 'reports/metrics.json',
            'reports/TEST_REPORT.md', 'reports/test_results.xml', 'tests/test_predictive_maint.py', 'docs/VERI_KAYNAGI.md',
            'scripts/build_notebook.py', 'scripts/run_notebook.py', 'DELIVERY_CHECKLIST.md', 'FINAL_REPORT.md'] + \
           [f'reports/figures/{f}' for f in ('01_sinif_dagilimi.png', '02_ozellik_dagilimlari.png', '03_arizayla_iliskili.png',
                                             '04_korelasyon.png', '05_karisiklik_matrisleri.png', '06_pr_egrileri.png',
                                             '07_esik_dengesi.png', '08_final_karisiklik.png', '09_ozellik_onemleri.png')]
missing = [f for f in REQUIRED if not (ROOT / f).is_file() or (ROOT / f).stat().st_size == 0]
check('Required files', not missing, f'{len(REQUIRED) - len(missing)}/{len(REQUIRED)} present' + (f'; missing {missing}' if missing else ''))

nb = nbformat.read(ROOT / 'notebooks/PredictiveMaint_AI.ipynb', as_version=4)
code = [c for c in nb.cells if c.cell_type == 'code']
errors = [o for c in code for o in c.get('outputs', []) if o.output_type == 'error']
src = '\n'.join(c.source for c in code)
check('Notebook executed, no errors', code and all(c.execution_count for c in code) and not errors, f'{len(code)} code cells executed, errors={len(errors)}')
need = ['stratify=y', 'random_state=SEED', 'StandardScaler', 'OneHotEncoder', 'Pipeline', 'LogisticRegression', 'RandomForestClassifier',
        'MLPClassifier', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'confusion_matrix', 'roc_auc_score',
        'average_precision_score', 'permutation_importance', 'X_val', "assert not set(LEAKY) & set(ENG)"]
miss = [x for x in need if x not in src]
check('Notebook contains required methods and leakage guard', not miss, 'split/encoding/scaling/3 models/all metrics/permutation importance/validation set/leakage assert' + (f'; MISSING {miss}' if miss else ''))
win = re.findall(r'(?<![A-Za-z])[A-Za-z]:(?:\\\\|/(?!/))', src)
check('No Windows-specific paths (Colab)', not win, 'none' if not win else str(win))

# numbers in the PDF must equal metrics.json
pdf = PdfReader(ROOT / 'reports/PredictiveMaint_AI_Rapor.pdf')
text = re.sub(r'\s+', ' ', ''.join(p.extract_text() for p in pdf.pages))
fin, th, sel = R['test_final'], R['threshold'], R['selected_model']
expect = {'rows': '10000 gözlem', 'failures': f"{R['data']['failures']} arıza", 'threshold': n(th['value'], 2),
          'final recall': n(fin['recall']), 'final precision': n(fin['precision']), 'final FNR': n(fin['fnr']),
          'final FN': f"{int(fin['fn'])} tanesi kaçırıldı", 'naive accuracy': n(R['naive_all_zero']['accuracy'], 4)}
for name, m in R['test_default'].items():
    expect[f'{name} PR-AUC'] = n(m['pr_auc']); expect[f'{name} recall'] = n(m['recall'])
for name, t in R['tuning'].items():
    expect[f'{name} val PR-AUC'] = n(t['val_pr_auc'], 4)
miss_pdf = {k: v for k, v in expect.items() if v not in text}
check('Report numbers == notebook metrics.json', not miss_pdf, f'{len(expect) - len(miss_pdf)}/{len(expect)} values found in PDF text' + (f'; MISSING {miss_pdf}' if miss_pdf else ''))
check('Report length 3-5 pages', 3 <= len(pdf.pages) <= 5, f'{len(pdf.pages)} pages')
kws = [k for k in ('sentetik', 'sızıntı', 'Yanlış negatif', 'nedensellik', 'akademik bir prototip', 'bağlantısı henüz yayımlanmadı') if k not in text]
check('Report states synthetic data, leakage, FN, non-causality, prototype, no fake link', not kws, 'all statements present' if not kws else f'MISSING {kws}')
readme = (ROOT / 'README.md').read_text(encoding='utf-8')
b5, s5 = R['test_default']['LogReg (temel)'], R['test_default'][sel]
rows_expected = [f"| {n(b5['precision'])} | {n(b5['recall'])} | {n(b5['f1'])} | {n(b5['pr_auc'])} | {int(b5['fn'])} |",
                 f"| {n(s5['precision'])} | {n(s5['recall'])} | {n(s5['f1'])} | {n(s5['pr_auc'])} | {int(s5['fn'])} |",
                 f"| **{n(fin['precision'])}** | **{n(fin['recall'])}** | **{n(fin['f1'])}** | {n(fin['pr_auc'])} | **{int(fin['fn'])}** |",
                 f"**{n(th['value'], 2)}**"]
miss_rm = [r for r in rows_expected if r not in readme]
check('README results table == metrics.json', not miss_rm, 'baseline, selected@0.5 and selected@threshold rows match' + (f'; MISSING {miss_rm}' if miss_rm else ''))
doc = Document(ROOT / 'reports/PredictiveMaint_AI_Rapor.docx')
auth = {doc.core_properties.author, doc.core_properties.last_modified_by, (pdf.metadata or {}).get('/Author'), nb.metadata['authors'][0]['name']}
check('Author metadata = Tugay Sinan Ergün only', auth == {AUTHOR}, str(auth))
check('Selected by validation, threshold from validation', sel == max(R['tuning'], key=lambda k: R['tuning'][k]['val_pr_auc']) and th['criterion'] == 'max F2 on validation',
      f"selected {sel} (val PR-AUC {R['tuning'][sel]['val_pr_auc']:.4f}); threshold {th['value']} ({th['criterion']})")
suite = ET.parse(ROOT / 'reports/test_results.xml').getroot()
suite = suite if suite.tag == 'testsuite' else suite.find('testsuite')
t, f, e = int(suite.get('tests')), int(suite.get('failures')), int(suite.get('errors'))
check('Automated tests', t >= 11 and f == 0 and e == 0, f'{t} tests, {f} failures, {e} errors')

if '--zip' in sys.argv:
    if ZIP.is_file():
        with zipfile.ZipFile(ZIP) as z:
            bad = z.testzip(); names = z.namelist()
        leaked = [x for x in names if re.search(r'(^|/)(\.venv|__pycache__|\.pytest_cache|\.ipynb_checkpoints)(/|$)|(^|/)\.env$', x)]
        miss_zip = [r for r in REQUIRED if f'PredictiveMaint-AI/{r}' not in names]
        check('ZIP integrity and contents', bad is None and not leaked and not miss_zip,
              f'{len(names)} files, {ZIP.stat().st_size / 1048576:.2f} MB, CRC ok={bad is None}, leaked={leaked or "none"}, missing={miss_zip or "none"}')
    else:
        check('ZIP integrity and contents', False, 'zip not found')

ok = all(r[1] for r in results)
lines = ['PredictiveMaint AI - Delivery Verification', f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}', '']
lines += [f'[{"PASS" if r[1] else "FAIL"}] {r[0]}: {r[2]}' for r in results]
lines += ['', f'OVERALL: {"PASS" if ok else "FAIL"} ({sum(r[1] for r in results)}/{len(results)})']
(ROOT / 'reports' / ('zip_verification.txt' if '--zip' in sys.argv else 'delivery_verification.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8'); print('\n'.join(lines)); sys.exit(0 if ok else 1)
