# import torch
# from torch.utils.data import DataLoader
# from model import ReportGenerator
# from train import ChestXrayDataset, collate_fn
# from transformers import AutoTokenizer
# from tqdm import tqdm
# import json
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def plot_and_save(data, title, xlabel, ylabel, filename):
#     plt.figure()
#     plt.hist(data, bins=30, alpha=0.7)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     plt.tight_layout()
#     os.makedirs('ChestXray/plots', exist_ok=True)
#     plt.savefig(f'ChestXray/plots/{filename}')
#     plt.close()

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tokenizer = AutoTokenizer.from_pretrained('t5-small')
#     # Only pass required arguments for text generation
#     ds = ChestXrayDataset('ChestXray/processed/test.jsonl', tokenizer, label_map_mesh=None, label_map_problems=None, transform=None)
#     dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
#     model = ReportGenerator(embed_dim=512, decoder_name='t5-small', num_labels=1).to(device)  # num_labels is a dummy value
#     # Load only matching weights (ignore classifier head)
#     state_dict = torch.load('ChestXray/best_model.pt', map_location=device)
#     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     bleu_scores_f, bleu_scores_i = [], []
#     rouge_scores_f, rouge_scores_i = [], []
#     sample_outputs = []
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     with torch.no_grad():
#         for batch in tqdm(dl):
#             img_f = batch['img_f'].to(device)
#             img_l = batch['img_l'].to(device)
#             # Generate findings
#             outputs_f, _ = model(img_f, img_l)
#             pred_f = tokenizer.decode(torch.argmax(outputs_f.logits, dim=-1)[0], skip_special_tokens=True)
#             # Generate impression
#             outputs_i, _ = model(img_f, img_l)
#             pred_i = tokenizer.decode(torch.argmax(outputs_i.logits, dim=-1)[0], skip_special_tokens=True)
#             # True
#             true_f = tokenizer.decode(batch['findings_input_ids'][0], skip_special_tokens=True)
#             true_i = tokenizer.decode(batch['impression_input_ids'][0], skip_special_tokens=True)
#             # BLEU
#             bleu_scores_f.append(sentence_bleu([true_f.split()], pred_f.split()))
#             bleu_scores_i.append(sentence_bleu([true_i.split()], pred_i.split()))
#             # ROUGE-L
#             rouge_scores_f.append(scorer.score(true_f, pred_f)['rougeL'].fmeasure)
#             rouge_scores_i.append(scorer.score(true_i, pred_i)['rougeL'].fmeasure)
#             # Print a few examples
#             if len(bleu_scores_f) <= 5:
#                 sample_outputs.append({
#                     'Findings True': true_f,
#                     'Findings Pred': pred_f,
#                     'Impression True': true_i,
#                     'Impression Pred': pred_i
#                 })
#     print(f'Findings BLEU: {np.mean(bleu_scores_f):.4f}, ROUGE-L: {np.mean(rouge_scores_f):.4f}')
#     print(f'Impression BLEU: {np.mean(bleu_scores_i):.4f}, ROUGE-L: {np.mean(rouge_scores_i):.4f}')
#     # Print sample outputs
#     for ex in sample_outputs:
#         print('---')
#         print('Findings True:', ex['Findings True'])
#         print('Findings Pred:', ex['Findings Pred'])
#         print('Impression True:', ex['Impression True'])
#         print('Impression Pred:', ex['Impression Pred'])
#     # Plot and save graphs
#     plot_and_save(bleu_scores_f, 'BLEU Score Distribution (Findings)', 'BLEU Score', 'Count', 'bleu_findings.png')
#     plot_and_save(bleu_scores_i, 'BLEU Score Distribution (Impression)', 'BLEU Score', 'Count', 'bleu_impression.png')
#     plot_and_save(rouge_scores_f, 'ROUGE-L Score Distribution (Findings)', 'ROUGE-L Score', 'Count', 'rouge_findings.png')
#     plot_and_save(rouge_scores_i, 'ROUGE-L Score Distribution (Impression)', 'ROUGE-L Score', 'Count', 'rouge_impression.png')

# if __name__ == '__main__':
#     main() 


# import torch
# from torch.utils.data import DataLoader
# from model import ReportGenerator
# from train import ChestXrayDataset, collate_fn
# from transformers import AutoTokenizer
# from tqdm import tqdm
# import json
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from torchvision import transforms  # <-- Added

# def plot_and_save(data, title, xlabel, ylabel, filename):
#     plt.figure()
#     plt.hist(data, bins=30, alpha=0.7)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     plt.tight_layout()
#     os.makedirs('ChestXray/plots', exist_ok=True)
#     plt.savefig(f'ChestXray/plots/{filename}')
#     plt.close()

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     tokenizer = AutoTokenizer.from_pretrained('t5-small')

#     # ✅ FIX: Add image transformation to convert PIL Image to Tensor
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # Converts PIL to Tensor
#     ])

#     ds = ChestXrayDataset(
#         'ChestXray/processed/test.jsonl',
#         tokenizer,
#         label_map_mesh=None,
#         label_map_problems=None,
#         transform=transform  # <-- Pass the transform here
#     )

#     dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     model = ReportGenerator(embed_dim=512, decoder_name='t5-small', num_labels=1).to(device)  # num_labels is a dummy value

#     # Load only matching weights (ignore classifier head)
#     state_dict = torch.load('ChestXray/best_model.pt', map_location=device)
#     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()

#     bleu_scores_f, bleu_scores_i = [], []
#     rouge_scores_f, rouge_scores_i = [], []
#     sample_outputs = []

#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

#     with torch.no_grad():
#         for batch in tqdm(dl):
#             img_f = batch['img_f'].to(device)
#             img_l = batch['img_l'].to(device)

#             # Generate findings
#             outputs_f, _ = model(img_f, img_l)
#             pred_f = tokenizer.decode(torch.argmax(outputs_f.logits, dim=-1)[0], skip_special_tokens=True)

#             # Generate impression
#             outputs_i, _ = model(img_f, img_l)
#             pred_i = tokenizer.decode(torch.argmax(outputs_i.logits, dim=-1)[0], skip_special_tokens=True)

#             # True
#             true_f = tokenizer.decode(batch['findings_input_ids'][0], skip_special_tokens=True)
#             true_i = tokenizer.decode(batch['impression_input_ids'][0], skip_special_tokens=True)

#             # BLEU
#             bleu_scores_f.append(sentence_bleu([true_f.split()], pred_f.split()))
#             bleu_scores_i.append(sentence_bleu([true_i.split()], pred_i.split()))

#             # ROUGE-L
#             rouge_scores_f.append(scorer.score(true_f, pred_f)['rougeL'].fmeasure)
#             rouge_scores_i.append(scorer.score(true_i, pred_i)['rougeL'].fmeasure)

#             # Print a few examples
#             if len(bleu_scores_f) <= 5:
#                 sample_outputs.append({
#                     'Findings True': true_f,
#                     'Findings Pred': pred_f,
#                     'Impression True': true_i,
#                     'Impression Pred': pred_i
#                 })

#     print(f'Findings BLEU: {np.mean(bleu_scores_f):.4f}, ROUGE-L: {np.mean(rouge_scores_f):.4f}')
#     print(f'Impression BLEU: {np.mean(bleu_scores_i):.4f}, ROUGE-L: {np.mean(rouge_scores_i):.4f}')

#     # Print sample outputs
#     for ex in sample_outputs:
#         print('---')
#         print('Findings True:', ex['Findings True'])
#         print('Findings Pred:', ex['Findings Pred'])
#         print('Impression True:', ex['Impression True'])
#         print('Impression Pred:', ex['Impression Pred'])

#     # Plot and save graphs
#     plot_and_save(bleu_scores_f, 'BLEU Score Distribution (Findings)', 'BLEU Score', 'Count', 'bleu_findings.png')
#     plot_and_save(bleu_scores_i, 'BLEU Score Distribution (Impression)', 'BLEU Score', 'Count', 'bleu_impression.png')
#     plot_and_save(rouge_scores_f, 'ROUGE-L Score Distribution (Findings)', 'ROUGE-L Score', 'Count', 'rouge_findings.png')
#     plot_and_save(rouge_scores_i, 'ROUGE-L Score Distribution (Impression)', 'ROUGE-L Score', 'Count', 'rouge_impression.png')

# if __name__ == '__main__':
#     main()

import torch
from torch.utils.data import DataLoader
from model import ReportGenerator
from train import ChestXrayDataset, collate_fn
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms  # <-- Needed for image to tensor

def plot_and_save(data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist(data, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('ChestXray/plots', exist_ok=True)
    plt.savefig(f'ChestXray/plots/{filename}')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    # Add required transform to convert PIL image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    ds = ChestXrayDataset(
        'ChestXray/processed/test.jsonl',
        tokenizer,
        label_map_mesh=None,
        label_map_problems=None,
        transform=transform
    )

    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = ReportGenerator(embed_dim=512, decoder_name='t5-small', num_labels=1).to(device)

    # Load pretrained weights
    state_dict = torch.load('ChestXray/best_model.pt', map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    bleu_scores_f, bleu_scores_i = [], []
    rouge_scores_f, rouge_scores_i = [], []
    sample_outputs = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Define start token ID (T5 uses pad_token as start)
    start_token_id = tokenizer.pad_token_id  # Usually 0 for T5

    with torch.no_grad():
        for batch in tqdm(dl):
            img_f = batch['img_f'].to(device)
            img_l = batch['img_l'].to(device)

            # Provide decoder_input_ids manually
            decoder_input_ids = torch.tensor([[start_token_id]]).to(device)

            # Generate findings
            outputs_f, _ = model(img_f, img_l, decoder_input_ids=decoder_input_ids)
            pred_f_ids = torch.argmax(outputs_f.logits, dim=-1)
            pred_f = tokenizer.decode(pred_f_ids[0], skip_special_tokens=True)

            # Generate impression
            outputs_i, _ = model(img_f, img_l, decoder_input_ids=decoder_input_ids)
            pred_i_ids = torch.argmax(outputs_i.logits, dim=-1)
            pred_i = tokenizer.decode(pred_i_ids[0], skip_special_tokens=True)

            # Ground truth
            true_f = tokenizer.decode(batch['findings_input_ids'][0], skip_special_tokens=True)
            true_i = tokenizer.decode(batch['impression_input_ids'][0], skip_special_tokens=True)

            # BLEU
            bleu_scores_f.append(sentence_bleu([true_f.split()], pred_f.split()))
            bleu_scores_i.append(sentence_bleu([true_i.split()], pred_i.split()))

            # ROUGE-L
            rouge_scores_f.append(scorer.score(true_f, pred_f)['rougeL'].fmeasure)
            rouge_scores_i.append(scorer.score(true_i, pred_i)['rougeL'].fmeasure)

            # Save sample output
            if len(bleu_scores_f) <= 5:
                sample_outputs.append({
                    'Findings True': true_f,
                    'Findings Pred': pred_f,
                    'Impression True': true_i,
                    'Impression Pred': pred_i
                })

    # Report metrics
    print(f'Findings BLEU: {np.mean(bleu_scores_f):.4f}, ROUGE-L: {np.mean(rouge_scores_f):.4f}')
    print(f'Impression BLEU: {np.mean(bleu_scores_i):.4f}, ROUGE-L: {np.mean(rouge_scores_i):.4f}')

    # Print sample predictions
    for ex in sample_outputs:
        print('---')
        print('Findings True:', ex['Findings True'])
        print('Findings Pred:', ex['Findings Pred'])
        print('Impression True:', ex['Impression True'])
        print('Impression Pred:', ex['Impression Pred'])

    # Save score distributions
    plot_and_save(bleu_scores_f, 'BLEU Score Distribution (Findings)', 'BLEU Score', 'Count', 'bleu_findings.png')
    plot_and_save(bleu_scores_i, 'BLEU Score Distribution (Impression)', 'BLEU Score', 'Count', 'bleu_impression.png')
    plot_and_save(rouge_scores_f, 'ROUGE-L Score Distribution (Findings)', 'ROUGE-L Score', 'Count', 'rouge_findings.png')
    plot_and_save(rouge_scores_i, 'ROUGE-L Score Distribution (Impression)', 'ROUGE-L Score', 'Count', 'rouge_impression.png')

if __name__ == '__main__':
    main()
