import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import datetime
from tqdm import tqdm
import numpy as np
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

from retriever_dataset import RetrieverDataset, cl_dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from model import ColBERT

from accelerate import Accelerator
import json

# from utils import set_seed

import wandb

from retriever_dataset import RetrieverDataset
from open_clip import create_model_from_pretrained, get_tokenizer
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ColBERT()
    # model.load_state_dict(torch.load("/root/nas/PythonCode/reranker/model/rad_iter_2_new_retrieve_top24_epoch_5_lr_0.0001_CL_False.pth", map_location='cuda:2'))# !!!!!

    print("load_pretrain")
    _, image_processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    print("finish load")

    device = "cuda:2"
    # device = "cpu"
    model.to(device)
    model.train()

    train_dataset = RetrieverDataset(args, tokenizer, image_processor, fold="train")

    bsz = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=bsz) 

    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    unfreeze_layers = ['text_linear', 'image_linear']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for unfreeze_layer in unfreeze_layers:
            if unfreeze_layer in name:
                print("unfreeze_layers", name)
                param.requires_grad = True
                break
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr)

    avg_kl_loss = 0.0
    record_avg_kl_loss = []
    for epoch in tqdm(range(args.num_epochs)):
        for index, (query_input_ids, query_attention_mask, query_image_input, key_input_ids, key_attention_mask, key_image_inputs, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            query_batch = query_input_ids.to(device)
            query_image_input = query_image_input.to(device)

            key_batch = key_input_ids.to(device)
            key_image_inputs = key_image_inputs.to(device)

            label = label.to(device)
            Pred = model(query_batch, query_image_input, key_batch, key_image_inputs, device)

            pred_dist = torch.log_softmax(Pred, dim=-1)
            gold_dist = torch.softmax(label * 5, dim=-1)

            loss = criterion(pred_dist, gold_dist)
            avg_kl_loss += loss
            if index % 50 == 0 and index != 0:
                record_avg_kl_loss.append(avg_kl_loss.item() / 50)
                print(f"history_avg_kl_loss: {record_avg_kl_loss}")
                print("avg_kl_loss:", avg_kl_loss / 50)
                avg_kl_loss = 0.0

            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), f'/root/nas/PythonCode/reranker/model/rad_iter_{str(args.iter_count)}_new_retrieve_top24_epoch_{str(args.num_epochs)}_lr_{str(args.lr)}.pth')

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="medqa_usmle_hf", choices=["medqa_usmle_hf", "strategyqa", "obqa"])
    parser.add_argument("--data_dir", type=str, default="/root/nas/PythonCode/medical_VQA/result") 
    parser.add_argument("--search_space_dir", type=str, default="") 
    parser.add_argument("--knowledge_base", type=str, default="wikipedia", choices=["wikipedia", "pubmed"])
    parser.add_argument("--save_dir", type=str, default="colbert_lr1e-3")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_cands", type=int, default=8, help="The number of negative samples")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--model_tau", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_report", action='store_true')
    parser.add_argument("--tau_annealing", action='store_true')
    parser.add_argument("--update_both", action='store_true')
    parser.add_argument("--loss_type", type=str, default="ce", choices=["kl", "ce"])
    parser.add_argument("--generate_search_space", action='store_true')
    parser.add_argument("--no_scheduler", action='store_true')
    parser.add_argument("--alpha", default=1.0, type=float)

    parser.add_argument("--only_from_rationale", action='store_true', help="coupled with combine candidates, but use candidate from r only")
    parser.add_argument("--only_from_question", action='store_true', help="coupled with combine candidates, but use candidate from q only")

    parser.add_argument("--iter_count", type=int, default=0)

    args = parser.parse_args()

    args.model_name = "/data/model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    print(f"Search Space: {args.search_space_dir}")

    args.tau_annealing = True
    args.update_both = True
    print(f"  [Default Setting] tau_annealing {args.tau_annealing}")
    print(f"  [Default Setting] update parameters from both query and key {args.update_both}")

    run(args)