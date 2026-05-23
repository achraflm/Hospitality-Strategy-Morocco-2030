import json
import sys

def extract_source(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        ct = c['cell_type']
        src = ''.join(c['source'])
        print(f'--- CELL {i} ({ct}) ---')
        print(src)
        print()

if __name__ == '__main__':
    extract_source(sys.argv[1])
