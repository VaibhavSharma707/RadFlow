import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Paths
REPORTS_CSV = 'ChestXray/indiana_reports.csv'
PROJ_CSV = 'ChestXray/indiana_projections.csv'
IMG_DIR = 'ChestXray/images/images_normalized'

# Load reports
reports = pd.read_csv(REPORTS_CSV)
projs = pd.read_csv(PROJ_CSV)

# Map uid to images by projection
def get_image_dict(projs):
    img_dict = {}
    for _, row in projs.iterrows():
        uid = str(row['uid'])
        if uid not in img_dict:
            img_dict[uid] = {}
        img_dict[uid][row['projection']] = os.path.join(IMG_DIR, row['filename'])
    return img_dict

img_dict = get_image_dict(projs)

data = []
for _, row in reports.iterrows():
    uid = str(row['uid'])
    images = img_dict.get(uid, {})
    frontal = images.get('Frontal', None)
    lateral = images.get('Lateral', None)
    if not frontal or not lateral:
        continue  # skip if missing either view
    entry = {
        'uid': uid,
        'frontal': frontal,
        'lateral': lateral,
        'findings': str(row.get('findings', '')),
        'impression': str(row.get('impression', '')),
        'MeSH': str(row.get('MeSH', '')),
        'Problems': str(row.get('Problems', '')),
    }
    data.append(entry)

# Split
data_train, data_test = train_test_split(data, test_size=0.15, random_state=42)
data_train, data_val = train_test_split(data_train, test_size=0.15, random_state=42)

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

os.makedirs('ChestXray/processed', exist_ok=True)
save_jsonl(data_train, 'ChestXray/processed/train.jsonl')
save_jsonl(data_val, 'ChestXray/processed/val.jsonl')
save_jsonl(data_test, 'ChestXray/processed/test.jsonl')

print(f"Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}") 