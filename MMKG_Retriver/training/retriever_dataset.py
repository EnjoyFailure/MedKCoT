import json
import os
import sys
sys.path.append('.')
import random
import argparse
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from PIL import Image

class RetrieverDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor, fold="train"):
        self.args = args

        with open(os.path.join(args.data_dir, "slake_train.json"), 'r') as f:
            self.dataset = json.load(f)
        self.image_processor = image_processor
    
        with open("/data/QA/Med-QA/Reranker/dataset/slake/e2image.json", 'r') as f:
            self.e2img = json.load(f)

        if args.debug:
            self.original_dataset = self.original_dataset[:100]

        self.dataset = [(_id, data) for _id, data in enumerate(self.dataset)]
        self.tokenizer = tokenizer
        self.max_seq_len = 512

        self.fold = fold
        self.k = args.n_cands if fold == "train" else 8
        self.n_k = 32
        
        os.makedirs(args.search_space_dir, exist_ok=True)
        self.search_space_dump = args.search_space_dir

    def preprocess(self, data, space, _idx):
        query = data["question"]

        query_image = Image.open(os.path.join("/root/nas/PythonCode/medical_VQA/result/slake_imgs", data["image_name"]))
        query_image_input = self.image_processor(query_image)

        keys = [item["chunk"] for item in space]

        query_outputs = self.tokenizer(query, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
        key_outputs = self.tokenizer(keys, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)

        key_images = [Image.open(os.path.join("/data/image-crawler/images/slake", I["entity"], self.e2img[I["entity"]])) for I in space]
        key_image_inputs = torch.stack([self.image_processor(key_image) for key_image in key_images])
        
        scores = [-item["ppl_norm2"] if item["score"] < 0 else item["ppl_norm2"] for index, item in enumerate(space)]

        query_input_ids = query_outputs["input_ids"]
        query_attention_mask = query_outputs["attention_mask"]
        key_input_ids = key_outputs["input_ids"]
        key_attention_mask = key_outputs["attention_mask"]
        score = torch.FloatTensor(scores)

        return query_input_ids, query_attention_mask, query_image_input, key_input_ids, key_attention_mask, key_image_inputs, score


    def __getitem__(self, idx):
        _idx, data = self.dataset[idx]
        load_path = os.path.join(self.search_space_dump, str(_idx).zfill(4) + ".json") 
        with open(load_path, 'r') as f:
            _ss = json.load(f)
        
        ss = _ss
        return self.preprocess(data, ss, _idx)

    def __len__(self):
        return len(self.dataset)
