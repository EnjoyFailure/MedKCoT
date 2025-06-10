from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
import numpy as np
from einops import rearrange

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from transformers import AutoTokenizer

from torch import nn
import torch.nn.functional as F



class ColBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.BioCLIP, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir="/root/.cache/huggingface/hub")
        self.text_linear = nn.Sequential(nn.Linear(512, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 512))

        self.image_linear = nn.Sequential(nn.Linear(512, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 512))


        self.criterion = NTXentLoss(temperature=0.5)
    

    def forward(self,
                queries: Dict[str, torch.LongTensor],
                query_image_input,
                documents: Dict[str, torch.LongTensor], 
                key_image_inputs,
                device):
        scores = []

        for query, query_image, document, key_image in zip(queries, query_image_input, documents, key_image_inputs):
            query_image = query_image.unsqueeze(0)

            query_vecs = self.BioCLIP.text(query)
            query_vecs= self.text_linear(query_vecs)
            query_img_vecs = self.BioCLIP.visual(query_image)
            query_img_vecs= self.image_linear(query_img_vecs)

            document_vecs = self.BioCLIP.text(document)
            document_vecs= self.text_linear(document_vecs)
            document_img_vecs = self.BioCLIP.visual(key_image)
            document_img_vecs = self.image_linear(document_img_vecs)

            text_score = torch.matmul(query_vecs, document_vecs.T)
            image_score = torch.matmul(query_img_vecs, document_img_vecs.T)

            score = 0.5 * text_score + 0.5 * image_score
            scores.append(score)

        scores = torch.stack(scores)
        return scores
    

    def get_features(self, image, text):
        visual_features = self.BioCLIP.visual(image)
        text_features = self.BioCLIP.text(text)

        return visual_features, text_features

    def get_text_features(self, text):
        text_features = self.BioCLIP.text(text)
        text_features= self.text_linear(text_features)

        return text_features
    

    def get_visual_features(self, image):
        visual_features = self.BioCLIP.visual(image)
        visual_features= self.image_linear(visual_features)

        return visual_features

