import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('honest-model-directory-oof-hub.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][1]['source'])
loc = {}
exec(src, loc)
df = loc['df']
print(f"Total community notebooks tracked: {len(df)}\n")

categories = {
    1: "Category 1: Pure Single Models",
    2: "Category 2: End-to-End Ensembles",
    3: "Category 3: OOF & Test Datasets"
}

for cat_id, title in categories.items():
    sub = df[df['Category'] == cat_id].sort_values(by='LB', ascending=False)
    print(f"=== {title} ({len(sub)} entries) ===")
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"{i:2d}. [LB: {r['LB']:.5f} | CV: {r['CV']:.5f} | {r['Folds']} Folds] {r['Notebook']} (by {r['Author']})")
        print(f"    URL: {r['Link']}")
    print()
