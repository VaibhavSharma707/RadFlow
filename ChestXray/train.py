import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import ReportGenerator
from transformers import AutoTokenizer
import numpy as np

def get_label_map(data, key):
    labels = set()
    for item in data:
        for l in item[key].split(';'):
            l = l.strip()
            if l:
                labels.add(l)
    return {l: i for i, l in enumerate(sorted(labels))}

class ChestXrayDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, label_map_mesh, label_map_problems, max_length=128, transform=None):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.label_map_mesh = label_map_mesh
        self.label_map_problems = label_map_problems
        self.max_length = max_length
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img_f = Image.open(s['frontal']).convert('RGB')
        img_l = Image.open(s['lateral']).convert('RGB')
        if self.transform:
            img_f = self.transform(img_f)
            img_l = self.transform(img_l)
        # Tokenize findings and impression
        findings = self.tokenizer(s['findings'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        impression = self.tokenizer(s['impression'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        out = {
            'img_f': img_f,
            'img_l': img_l,
            'findings_input_ids': findings['input_ids'].squeeze(0),
            'findings_attention_mask': findings['attention_mask'].squeeze(0),
            'impression_input_ids': impression['input_ids'].squeeze(0),
            'impression_attention_mask': impression['attention_mask'].squeeze(0),
        }
        # Only add multi-label if label maps are provided and not None
        if self.label_map_mesh is not None and self.label_map_problems is not None:
            mesh_vec = np.zeros(len(self.label_map_mesh), dtype=np.float32)
            for l in s['MeSH'].split(';'):
                l = l.strip()
                if l and l in self.label_map_mesh:
                    mesh_vec[self.label_map_mesh[l]] = 1.0
            prob_vec = np.zeros(len(self.label_map_problems), dtype=np.float32)
            for l in s['Problems'].split(';'):
                l = l.strip()
                if l and l in self.label_map_problems:
                    prob_vec[self.label_map_problems[l]] = 1.0
            out['mesh'] = torch.tensor(mesh_vec)
            out['problems'] = torch.tensor(prob_vec)
        return out

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k.startswith('img_'):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    # Load data
    train_data = [json.loads(l) for l in open('ChestXray/processed/train.jsonl')]
    val_data = [json.loads(l) for l in open('ChestXray/processed/val.jsonl')]
    label_map_mesh = get_label_map(train_data, 'MeSH')
    label_map_problems = get_label_map(train_data, 'Problems')
    # Save label maps for evaluation
    with open('ChestXray/processed/label_map_mesh.json', 'w') as f:
        json.dump(label_map_mesh, f)
    with open('ChestXray/processed/label_map_problems.json', 'w') as f:
        json.dump(label_map_problems, f)
    # Datasets
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = ChestXrayDataset('ChestXray/processed/train.jsonl', tokenizer, label_map_mesh, label_map_problems, transform=transform)
    val_ds = ChestXrayDataset('ChestXray/processed/val.jsonl', tokenizer, label_map_mesh, label_map_problems, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    # Model
    model = ReportGenerator(embed_dim=512, decoder_name='t5-small', num_labels=len(label_map_mesh)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    for epoch in range(10):
        model.train()
        pbar = tqdm(train_dl, desc=f'Epoch {epoch+1} [train]')
        for batch in pbar:
            optimizer.zero_grad()
            img_f = batch['img_f'].to(device)
            img_l = batch['img_l'].to(device)
            # Findings
            outputs_f, mesh_logits = model(img_f, img_l, decoder_input_ids=batch['findings_input_ids'].to(device), labels=batch['findings_input_ids'].to(device))
            # Impression
            outputs_i, _ = model(img_f, img_l, decoder_input_ids=batch['impression_input_ids'].to(device), labels=batch['impression_input_ids'].to(device))
            # Multi-label (MeSH)
            mesh_loss = loss_fn(mesh_logits, batch['mesh'].to(device))
            # Total loss
            loss = outputs_f.loss + outputs_i.loss + mesh_loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f'Epoch {epoch+1} [val]'):
                img_f = batch['img_f'].to(device)
                img_l = batch['img_l'].to(device)
                outputs_f, mesh_logits = model(img_f, img_l, decoder_input_ids=batch['findings_input_ids'].to(device), labels=batch['findings_input_ids'].to(device))
                outputs_i, _ = model(img_f, img_l, decoder_input_ids=batch['impression_input_ids'].to(device), labels=batch['impression_input_ids'].to(device))
                mesh_loss = loss_fn(mesh_logits, batch['mesh'].to(device))
                loss = outputs_f.loss + outputs_i.loss + mesh_loss
                val_loss += loss.item()
        val_loss /= len(val_dl)
        print(f'Epoch {epoch+1} val_loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ChestXray/best_model.pt')
            print('Saved best model.')

if __name__ == '__main__':
    main() 