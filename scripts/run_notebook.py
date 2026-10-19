"""Execute notebooks/PredictiveMaint_AI.ipynb in place (from the project folder) and fail on any cell error.
Run: .venv/Scripts/python scripts/run_notebook.py"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / 'notebooks' / 'PredictiveMaint_AI.ipynb'
nb = nbformat.read(path, as_version=4)
t = time.time()
NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
nbformat.write(nb, path)
errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
print(f'executed {sum(c.cell_type == "code" for c in nb.cells)} code cells in {time.time() - t:.1f}s, errors: {len(errors)}')
sys.exit(1 if errors else 0)
