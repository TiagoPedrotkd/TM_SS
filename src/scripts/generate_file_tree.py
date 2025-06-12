import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "file_tree.json"

def build_tree(directory, base):
    tree = {}
    for entry in sorted(os.listdir(directory)):
        path = directory / entry
        if path.is_dir():
            tree[entry] = build_tree(path, base)
        else:
            # Caminho relativo à pasta results
            rel_path = path.relative_to(base).as_posix()
            tree[entry] = rel_path
    return tree

if __name__ == "__main__":
    tree = build_tree(RESULTS_DIR, RESULTS_DIR)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    print(f"Arquivo {OUTPUT_FILE} gerado com sucesso.")