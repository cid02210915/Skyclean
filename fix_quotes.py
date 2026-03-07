"""Fix curly/smart quotes in Python source files, replacing them with straight ASCII quotes."""

import os
import sys

CURLY_QUOTES = {
    '\u201c': '"',  # left double quotation mark
    '\u201d': '"',  # right double quotation mark
    '\u2018': "'",  # left single quotation mark
    '\u2019': "'",  # right single quotation mark
}

def fix_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        original = f.read()
    fixed = original
    for curly, straight in CURLY_QUOTES.items():
        fixed = fixed.replace(curly, straight)
    if fixed != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"Fixed: {path}")
    else:
        print(f"No curly quotes found: {path}")

if __name__ == "__main__":
    targets = sys.argv[1:] if sys.argv[1:] else ["skyclean/ml/data.py"]
    for target in targets:
        if os.path.isfile(target):
            fix_file(target)
        elif os.path.isdir(target):
            for root, _, files in os.walk(target):
                for fname in files:
                    if fname.endswith(".py"):
                        fix_file(os.path.join(root, fname))
        else:
            print(f"Not found: {target}")
