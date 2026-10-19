"""Create dist/PredictiveMaint_AI_Teslim.zip (top folder PredictiveMaint-AI/). Excludes virtual envs, caches, secrets and old dists.
Run: .venv/Scripts/python scripts/package.py
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'dist' / 'PredictiveMaint_AI_Teslim.zip'
EXCLUDE = re.compile(r'(^|/)(\.venv|__pycache__|\.pytest_cache|\.ipynb_checkpoints|dist|\.git)(/|$)|(^|/)\.env$|\.pyc$')


def main():
    OUT.parent.mkdir(exist_ok=True)
    files = [p for p in sorted(ROOT.rglob('*')) if p.is_file() and not EXCLUDE.search(p.relative_to(ROOT).as_posix())]
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, 'PredictiveMaint-AI/' + p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
    print(f'{OUT}: {len(files)} files, {OUT.stat().st_size / 1048576:.2f} MB, CRC {"OK" if bad is None else "BAD " + bad}')


if __name__ == '__main__':
    main()
