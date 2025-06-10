import os
import sys
sys.path.append("/data/KG/HuatuoGPT-Vision")

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64
import json
from langchain.prompts import PromptTemplate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from llava.model.builder import load_pretrained_model
from tqdm import tqdm
from open_clip import create_model_from_pretrained
from PIL import Image
from open_clip import create_model_from_pretrained
from collections import defaultdict
from model2 import ColBERT
from cli import HuatuoChatbot
import time
from requests.exceptions import Timeout


os.environ['OPENAI_API_KEY'] = "xxx"
os.environ['OPENAI_API_BASE'] = "xxx"

conv_mode = "vicuna_v1"
temperature = 0.0
top_p = None
num_beams = 1

biomed_model, biomed_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomed_model.eval()

biomed_tokenizer = AutoTokenizer.from_pretrained("/data/model/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

biomed_model = ColBERT()
biomed_model.load_state_dict(torch.load("/root/nas/PythonCode/reranker/model/retrieve_top24_epoch_5_lr_0.0001.pth", map_location='cuda:0'))


model_name="gpt-4o"

KG_path = "/data/KG/SLAKE/knowledge_graph.json"
with open(KG_path, "r") as file:
    kg_datas = json.load(file)


qa_train_path = "/data/KG/SLAKE/train.json"
with open(qa_train_path, "r") as file:
    qa_datas = json.load(file)


entity2related_path = "/data/KG/SLAKE/json/entity2related_entities.json"
with open(entity2related_path, "r") as file:
    entity2related_datas = json.load(file)

def llm(input, image_data, timeout):
    gpt = ChatOpenAI(model_name=model_name)
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text",
         "text": input},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpg;base64,{encoded_image}"}}
    ])
    response = gpt.invoke([message], timeout=timeout)
    llm_name = response.response_metadata["model_name"]
    return response, llm_name


def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words



def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=-1)


def process_question_and_image(question, answer, image_path):
    question_tokens = biomed_tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=256)
    answer_tokens = biomed_tokenizer(answer, return_tensors='pt', padding=True, truncation=True, max_length=256)
    question_image = Image.open(image_path)
    question_image_input = biomed_preprocess(question_image).unsqueeze(0)
    return question_tokens, answer_tokens, question_image_input

def process_knowledge_graph(knowledge_graph):
    biomed_model.eval()

    entity_text_embeddings = []
    entity_image_embeddings = []
    
    target_dim = 512  

    with torch.no_grad():
        for entity in knowledge_graph:
            entity_chunk = entity["chunk"]
      
            entity_text_tokens = biomed_tokenizer(entity_chunk, return_tensors='pt', padding=True, truncation=True, max_length=256)
            entity_text_embedding = biomed_model.encode_text(entity_text_tokens['input_ids']).detach()

            if entity_text_embedding.shape[1] != target_dim:
                if entity_text_embedding.shape[1] < target_dim:
                    padding = torch.zeros((entity_text_embedding.shape[0], target_dim - entity_text_embedding.shape[1]))
                    entity_text_embedding = torch.cat([entity_text_embedding, padding], dim=1)
                else:
                    entity_text_embedding = entity_text_embedding[:, :target_dim]  

            if "image" in entity:
                entity_image_path = os.path.join("/data/KG/slake_img", entity["image"])
                
                if os.path.exists(entity_image_path):
                    entity_image = Image.open(entity_image_path)
                    entity_image_input = biomed_preprocess(entity_image).unsqueeze(0)
                    entity_image_embedding = biomed_model.encode_image(entity_image_input).detach()

                    if entity_image_embedding.shape[1] != target_dim:
                        if entity_image_embedding.shape[1] < target_dim:
                            padding = torch.zeros((entity_image_embedding.shape[0], target_dim - entity_image_embedding.shape[1]))
                            entity_image_embedding = torch.cat([entity_image_embedding, padding], dim=1)
                        else:
                            entity_image_embedding = entity_image_embedding[:, :target_dim] 

                    entity_image_embeddings.append(entity_image_embedding)
                else:
                    print(f"Warning: Image file not found: {entity_image_path}")
                    entity_image_embeddings.append(torch.zeros((1, target_dim)))
            else:

                entity_image_embeddings.append(torch.zeros((1, target_dim))) 

            entity_text_embeddings.append(entity_text_embedding)
    
    return torch.cat(entity_text_embeddings), torch.cat(entity_image_embeddings)

def get_idx_from_kgs_on_name(name):
    for index, entity in enumerate(kg_datas):
        if entity["entity"] == name:
            return index
    
    return None

def rerank_with_multimodal(question_tokens, answer_tokens, question_image_input, all_indices, knowledge_graph, entity_text_embeddings, entity_image_embeddings, entity2related_datas):
    question_similarities = []
    answer_similarities = []
    image_similarities = []
    total_similarities = []
    total_adj_similarities = []
    

    for idx in all_indices:
        entity_text_embedding = entity_text_embeddings[idx]
        entity_image_embedding = entity_image_embeddings[idx]
        question_embedding = biomed_model.encode_text(question_tokens['input_ids']).detach()
        answer_embedding = biomed_model.encode_text(answer_tokens['input_ids']).detach()
        image_embedding = biomed_model.encode_image(question_image_input).detach()

        question_similarity = cosine_similarity(question_embedding, entity_text_embedding).item()
        answer_similarity = cosine_similarity(answer_embedding, entity_text_embedding).item()
        image_similarity = cosine_similarity(image_embedding, entity_image_embedding).item()
        
        total_similarity = question_similarity + answer_similarity + image_similarity
        
        question_similarities.append(question_similarity)
        answer_similarities.append(answer_similarity)
        image_similarities.append(image_similarity)
        total_similarities.append(total_similarity)

        entity = knowledge_graph[idx]
        adj_entities = entity2related_datas.get(entity["entity"])
        
        # consider adj entity
        total_adj_similarity = 0
        if adj_entities is not None:
            adj_length = len(adj_entities)
            if adj_length != 0:
                y = 0.2

                for adj_e in adj_entities:
                    adj_index = get_idx_from_kgs_on_name(adj_e)
                    if adj_index is None:
                        continue

                    adj_question_similarity = cosine_similarity(question_embedding, entity_text_embeddings[adj_index]).item()
                    adj_answer_similarity = cosine_similarity(answer_embedding, entity_text_embeddings[adj_index]).item()
                    adj_image_similarity = cosine_similarity(image_embedding, entity_image_embeddings[adj_index]).item()
                    
                    total_adj_similarity += adj_question_similarity + adj_answer_similarity + adj_image_similarity
                
                total_adj_similarity = y * (total_adj_similarity / adj_length)
        
        total_adj_similarities.append(total_similarity + total_adj_similarity)
    
    sorted_indices = sorted(range(len(total_adj_similarities)), key=lambda i: total_adj_similarities[i], reverse=True)
    top_m_indices = sorted_indices[:3]

    final_sorted_indices = [all_indices[i] for i in sorted_indices]
    
    final_top_m_indices = [all_indices[i] for i in top_m_indices]
    
    return final_sorted_indices, final_top_m_indices, [total_adj_similarities[i] for i in top_m_indices]


template = """
You are a medical expert. Based on the following input, generate a **strictly forward-reasoned** chain-of-thought explanation. 

Absolutely **do not** mention or imply that the answer is already known.  
Prohibited examples (do NOT appear in the output):
- “Since the answer is known to be X, we can assume...”
- “Given the provided answer, it makes sense that...”
- “Because the answer is Y…”

Proper reasoning should:
- Be **strictly derived from the question, image, and retrieved knowledge**, not by verifying or justifying a known answer.
- Treat the answer as the natural conclusion of your reasoning, **not as an input**.
- Avoid any meta-language such as “the correct answer is,” “as given,” etc.

Follow this five-step template exactly:
1. Identify key question elements:  
   - Extract the critical symptoms or descriptions from the question, without assuming anything about the answer.  
2. Image analysis:  
   - Specify the imaging modality, projection, and orientation (e.g., “This is a posterior–anterior chest X-ray”).  
   - Point out exactly which anatomical structures are visible and what you observe (e.g., “cardiac silhouette, lung fields, costophrenic angles”).  
   - Describe every supportive marker in detail (e.g., “the heart border is rounded and extends beyond the mid-clavicular line, cardiothoracic ratio > 0.5; lung fields are uniformly radiolucent, with no focal consolidations or nodules”).  
3. Based on medical knowledge analysis (Retrieved knowledge, selective use only):
   - Integrate only the knowledge **directly relevant** to the image findings and clinical question;  
   - Do **not** mention that the knowledge was “retrieved” or “extracted” — instead, **embed** the relevant knowledge points naturally into the reasoning;  
   - Use this knowledge to **support, contrast, or exclude** possible diagnoses.  
   * Example: “An enlarged cardiac silhouette without pulmonary abnormalities is often associated with conditions like dilated cardiomyopathy.”  
   * Avoid: “According to the retrieved knowledge, cardiomegaly is linked to…”   
4. Synthesize and Conclude with Exact Answer:  
   - Explicitly mention the observed image features (e.g., enlarged cardiac silhouette);
   - Use medical terminology to justify the exclusion of other possibilities;
   - Clearly state the final diagnostic answer to the question based on reasoning, not speculation.

Gold Example:
Input Example:
{{
  "qid": "example_001",
  "question": "Is the abnormality in the heart or the lungs?",
  "answer": "Heart"
}}
Output Example:
{{
  "qid": "example_001",
  "chain_of_thought": \"""
    1. Identify key question elements: The question asks whether the observed abnormality is located in the heart or lungs. No specific symptoms are given, so image and clinical knowledge must guide reasoning.
    2. Image analysis: This is a posterior–anterior chest X-ray. Visible structures include the cardiac silhouette, lung fields, and costophrenic angles. The cardiac silhouette appears rounded and extends beyond the mid-clavicular lines bilaterally, with a cardiothoracic ratio exceeding 0.5, indicating cardiomegaly. Lung fields are uniformly radiolucent, without focal opacities, nodules, or pleural effusions.
    3. Based on medical knowledge analysis: A cardiothoracic ratio exceeding 0.5 is usually associated with an enlarged heart. Therefore, the morphology of the cardiac shadow in this image suggests that the lesion is located in the heart.  
    4. Synthesize and Conclude with Exact Answer: Integrating the clear evidence of an enlarged cardiac silhouette (CTR > 0.5) and absence of pulmonary parenchymal abnormalities, the abnormality is conclusively in the heart.
    \"""
}}

Input:
{{
    "qid": {qid},
    "question": {question},
    "answer": {answer},
}}

Relevant medical knowledge:
{retrieved_chunks}

Output Requirements:
```json
{{
    "qid": QID,
    "chain_of_thought": \"""
        1. Identify key question elements: …
        2. Image analysis: …
        3. Based on medical knowledge analysis: …
        4. Synthesize and Conclude with Exact Answer: …
        \""" 
}}
"""
prompt = PromptTemplate(
    input_variables = ["qid", "question", "answer", "retrieved_chunks"],
    template=template,
)

entity_text_embeddings, entity_image_embeddings =  process_knowledge_graph(kg_datas)

image_base_path = "/data/KG/SLAKE/imgs"
processed_index = -1
for index_qa, qa in enumerate(tqdm(qa_datas)):
    if index_qa <= processed_index:
        continue

    qid = qa["qid"]
    question = qa["question"]
    answer = qa["answer"]
    image_path = qa["img_name"] 
    
    question_tokens, answer_tokens, question_image_input = process_question_and_image(question, answer, os.path.join(image_base_path, image_path))
    
    all_indices = []
    for i in range(len(kg_datas)):
        all_indices.append(i)
    sorted_indices, top_3_indices, top_3_similarities = rerank_with_multimodal(question_tokens, answer_tokens, question_image_input, all_indices, kg_datas, entity_text_embeddings, entity_image_embeddings, entity2related_datas)

    retrieved_chunks_with_similarities = [
        {
            "entity": kg_datas[idx]['entity'],
            "chunk": kg_datas[idx]['chunk'],
            "similarity": similarity
        }
        for idx, similarity in zip(top_3_indices, top_3_similarities)
    ]
    

    retrieved_chunks = "\n".join([chunk_info["chunk"] for chunk_info in retrieved_chunks_with_similarities])

    complete_prompt = prompt.format(qid=qid, question=question, answer=answer, retrieved_chunks=retrieved_chunks)
    # print(complete_prompt)

    complete_image_path = os.path.join(image_base_path, image_path)
    with open(complete_image_path, "rb") as file:
        image = file.read()

   
    while True:
        try:
    
            response, model_name = llm(input=complete_prompt, image_data=image, timeout=15)
            # print(response.content)
            time.sleep(2)
            break 
        except Timeout:
            print("time out")
            continue
        except Exception as e: 
            print(f"{e}")
            continue 
    
    file_path = '/root/nas/PythonCode/medical_VQA/result/gpt_reason_QA_slake_with_KG.json'
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4) 


    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    results.append({
        "qid": qid,
        "response": response.content
    })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4) 