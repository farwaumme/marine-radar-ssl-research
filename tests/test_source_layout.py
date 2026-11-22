"""Dependency-free static smoke check; does not execute models."""
from pathlib import Path
import ast
root = Path(__file__).resolve().parents[1]
paths = sorted((root / 'src').rglob('*.py'))
assert len(paths) >= 15, f'Expected >=15 Python scripts, got {len(paths)}'
for path in paths:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'PASS: {len(paths)} Python files parsed successfully')
