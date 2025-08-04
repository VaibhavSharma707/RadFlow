import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DualImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Use pretrained ResNet for both images
        self.encoder_frontal = models.resnet34(weights='IMAGENET1K_V1')
        self.encoder_lateral = models.resnet34(weights='IMAGENET1K_V1')
        self.encoder_frontal.fc = nn.Linear(self.encoder_frontal.fc.in_features, embed_dim)
        self.encoder_lateral.fc = nn.Linear(self.encoder_lateral.fc.in_features, embed_dim)
    def forward(self, img_frontal, img_lateral):
        feat_f = self.encoder_frontal(img_frontal)
        feat_l = self.encoder_lateral(img_lateral)
        return torch.cat([feat_f, feat_l], dim=1)  # (B, 2*embed_dim)

class ReportGenerator(nn.Module):
    def __init__(self, embed_dim=512, decoder_name='t5-small', num_labels=100):
        super().__init__()
        self.img_encoder = DualImageEncoder(embed_dim)
        self.decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_name)
        self.proj = nn.Linear(2*embed_dim, self.decoder.config.d_model)
        self.classifier = nn.Linear(2*embed_dim, num_labels)  # For multi-label
    def forward(self, img_frontal, img_lateral, decoder_input_ids=None, labels=None):
        img_feat = self.img_encoder(img_frontal, img_lateral)  # (B, 2*embed_dim)
        dec_inputs = self.proj(img_feat).unsqueeze(1)  # (B, 1, d_model)
        # Use encoder_outputs for T5
        outputs = self.decoder(
            encoder_outputs=(dec_inputs,),
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
        class_logits = self.classifier(img_feat)
        return outputs, class_logits 