# ChestXray Radiology Report Generation

This project trains a deep learning model to generate structured radiology reports (findings, impression, MeSH/Problems) from chest X-ray images (frontal and lateral views).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the data:
   - Place the `indiana_reports.csv`, `indiana_projections.csv`, and normalized images in the `ChestXray/` directory as structured.

## Preprocessing

Run the preprocessing script to create train/val/test splits:
```bash
python preprocess.py
```

## Training

Train the model:
```bash
python train.py
```
- The best model will be saved as `ChestXray/best_model.pt`.

## Evaluation

Evaluate the model on the test set:
```bash
python evaluate.py
```
- Prints BLEU/ROUGE scores for findings and impression, and F1 for MeSH multi-label classification.

## Inference

To generate a report for new images, load the model and tokenizer from `model.py` and use the same preprocessing as in the dataset.

---

**Note:**
- GPU recommended for training.
- The model uses ResNet encoders and a T5 decoder (HuggingFace Transformers). 