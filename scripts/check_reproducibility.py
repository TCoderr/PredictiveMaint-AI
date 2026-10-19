"""Re-run the notebook from scratch and confirm reports/metrics.json is identical (training time excluded).
Writes reports/reproducibility_check.txt. Run: .venv/Scripts/python scripts/check_reproducibility.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'reports' / 'metrics.json'
before = json.loads(M.read_text(encoding='utf-8'))
run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'run_notebook.py')], capture_output=True, text=True, encoding='utf-8', errors='replace')
after = json.loads(M.read_text(encoding='utf-8'))
t1, t2 = before.pop('training_seconds', None), after.pop('training_seconds', None)
same = before == after
diff = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
text = (f'PredictiveMaint AI - reproducibility check\nGenerated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
        f'Notebook re-run exit code: {run.returncode} | {run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""}\n'
        f'metrics.json identical (excluding training_seconds {t1} -> {t2}): {same}\n' + (f'differing keys: {diff}\n' if diff else ''))
(ROOT / 'reports' / 'reproducibility_check.txt').write_text(text, encoding='utf-8')
print(text)
sys.exit(0 if same and run.returncode == 0 else 1)
