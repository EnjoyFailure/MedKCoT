import json
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_openai.llms import OpenAI
import base64
import time
from requests.exceptions import Timeout
import math
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['OPENAI_API_KEY'] = "xxx"
os.environ['OPENAI_API_BASE'] = "xxx"



def llm(gpt, input, image_data, timeout):
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text",
         "text": input},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpg;base64,{encoded_image}"}}
    ])
    response = gpt.invoke([message], timeout=timeout)
    # print(response.json())
    llm_name = response.response_metadata["model_name"]
    return response, llm_name


def get_token_logprobs(response):
    logprobs = response.response_metadata["logprobs"]["content"]
    return logprobs

def get_qa_information(qa_datas, qid):
    for qa in qa_datas:
        if qa["qid"] == qid:
            return qa["question"], qa["answer"], qa["image_name"]

def get_kg_information(kg_datas, entity):
    for kg in kg_datas:
        if kg["entity"] == entity:
            return kg["chunk"]


def get_old_information(old_top24, entity):
    for old in old_top24:
        if old["entity"] == entity:
            return old

#
def computer_pll(response, logprobs, n):
    scores = []

    for token_dict in logprobs:
        if token_dict["token" ]== "true" or token_dict["token"]== " true":
                scores.append(math.exp(token_dict["logprob"]))
        if token_dict["token"] == "false" or token_dict["token"] == " false":
            scores.append(-math.exp(token_dict["logprob"]))

  
    ppl_norm1_list = []
    ppl_norm2_list = []
    for i in range(n):
        # start_marker = f"[start{i+1}]"
        # end_marker = f"[end{i+1}]"
        start_marker = f"start{i+1}"
        end_marker = f"end{i+1}"
        start_index = response.find(start_marker)
        end_index = response.find(end_marker)
        if start_index != -1 and end_index != -1 and end_index > start_index:
            char_count = 0
            start_token_idx = end_token_idx = None
            for j, token in enumerate(logprobs):
                char_count += len(token["token"])
                if start_token_idx is None and char_count >= start_index + len(start_marker):
                    start_token_idx = j
                if end_token_idx  is None and char_count >= end_index:
                    end_token_idx = j
                    break
            
    
            target_logprobs = []
            for token in logprobs[start_token_idx:end_token_idx+1]:
                target_logprobs.append(token["logprob"])
            
            ppl = np.exp(-np.sum(target_logprobs) / len(target_logprobs))

            norm_ppl1 =  1 / (1 + np.log(ppl)) 
            norm_ppl2 = 1 / ppl
            ppl_norm1_list.append(norm_ppl1)
            ppl_norm2_list.append(norm_ppl2)

        else:
            print(response)
            raise ValueError("check error")
    
    return scores, ppl_norm1_list, ppl_norm2_list

relative_template = """
You are a medical imaging diagnosis expert. Analyze the correlation between the given medical entity and QA pair based on the following multimodal criteria:

Role: 
- As a senior physician specializing in clinical diagnosis
- Combine medical image metadata with textual descriptions
- Make definitive yes/no judgments

QA Pair Attributes: 
{{
    "qid": {qid},
    "question": {question},
    "answer": {answer},
}}
Entity Attributes:
{{
    "entity_list": {entity_list},
    "chunk_list": {chunk_list},
}}
Note: A QA pair corresponds to a set of entity information, and the relevance of this QA pair for each entity information should be judged.

Output Requirements:
```json1
{{
    "qid": QID1,
    "entity": "entity_name_1",
    "is_related": "true/false",
    "reason": [start1]"Generated reasoning"[end1]# Please provide a concise reasoning rather than an extended one.
}}
```
...
Note: 
- You only need to output this json format, you do not need to output any additional content.
- Mandatory markers: You must include the exact markers [start1], [end1], [start2], [end2], etc., in the output.
- Be concise: Keep reasoning short and within the marked section.

"""
relative_prompt = PromptTemplate(
    input_variables = ["entity_list", "chunk_list", "qid", "question", "answer"],
    template=relative_template,
)

KG_path = "/data/KG/SLAKE/knowledge_graph.json"
with open(KG_path, "r") as file:
    kg_datas = json.load(file)

qa_train_path = "/root/nas/PythonCode/medical_VQA/result/slake_train.json"
with open(qa_train_path, "r") as file:
    qa_datas = json.load(file)


new_retriver_top24 = "/root/nas/PythonCode/medical_VQA/result/slake_trained_retriver_top24_iters/top_24_iter_0_new.json"
with open(new_retriver_top24, "r") as f:
    new_24_datas = json.load(f)


old_folder_path = "/root/nas/PythonCode/medical_VQA/result/slake_ppls/sorted_ppl_2_top24"
old_retriver_top24 = sorted(os.listdir(old_folder_path))

image_base_path = "/root/nas/PythonCode/medical_VQA/result/slake_imgs"

def worker(index, old_top24_json=None):
    model_name="gpt-4o"
    gpt = ChatOpenAI(model_name=model_name).bind(logprobs=True)
    with open(os.path.join(old_folder_path, old_top24_json)) as f:
        old_top24 = json.load(f)
    
    old_entities = []
    for old in old_top24:
        old_entities.append(old["entity"])
    
    new_top24= new_24_datas[index]

    new_entities = new_top24["top_24_entities"]

    save_result = []
    repetitive_count = 0
    for new_entity in new_entities:
        if new_entity not in old_entities:
            qid = new_24_datas[index]["qid"]
            entity = new_entity
            question, answer, img_name = get_qa_information(qa_datas=qa_datas, qid=qid)
            chunk = get_kg_information(kg_datas=kg_datas, entity=entity)
            question_and_answer = f"question: {question} answer: {answer}"

            prompt = relative_prompt.format(entity_list=[entity], chunk_list=[chunk], qid=qid, question=question, answer=answer)
            # print(prompt)

            image_path = os.path.join(image_base_path, img_name)
            with open(image_path, "rb") as file:
                image = file.read()
    
            while True:
                try:
                    response, model_name = llm(gpt=gpt, input=prompt, image_data=image, timeout=10)
                    logprobs = get_token_logprobs(response=response)
                    # print(response.content)
                    # time.sleep(2)
                    break 
                except Timeout:
                    print("time out")
                    time.sleep(2)
                    continue
                except Exception as e: 
                    print(f"error")
                    time.sleep(2)
                    continue  
            
            try:
                scores, ppl_norm1_list, ppl_norm2_list = computer_pll(response=response.content, logprobs=logprobs, n=1)
            except Exception as e:
                print(f"{e}")
                continue
            score = scores[0]
            ppl_norm1 = ppl_norm1_list[0]
            ppl_norm2 = ppl_norm2_list[0]

            final_data = {
                "qid": qid,
                "entity": entity,
                "score": score,
                "ppl_norm1": ppl_norm1,
                "ppl_norm2": ppl_norm2,
                "img_name": img_name,
                "question": question_and_answer,
                "chunk": chunk
            }
            save_result.append(final_data)
            
        else:
            repetitive_count += 1
            final_data = get_old_information(old_top24, new_entity)
            save_result.append(final_data)
    
    save_path = "/root/nas/PythonCode/medical_VQA/result/slake_ppls/ppl_2_top24/" + str(index).zfill(4) + ".json"
    if not os.path.exists(save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    with open(save_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, ensure_ascii=False, indent=4)
    
    return index, repetitive_count

processed_index = -1
datas = old_retriver_top24
max_workers = 16

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(worker, index, old_top24_json)
               for index, old_top24_json in enumerate(datas)
               if index > processed_index]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        try:
            idx, rep_cnt = future.result()
            print(f"Index {idx:04d} repetitive count: {rep_cnt}")
        except Exception as e:
            print(f"[worker  error] index: {idx}. {e}")