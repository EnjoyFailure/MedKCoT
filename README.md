# MedKCoT

This is the official github repository for the paper "MedKCoT: A Multi-modal Knowledge Graph based Chain of Thought Generation framework for Medical Visual Question Answering".

We present the source code.

## Contents

- [MedKCoT](#MedKCoT)
  - [Contents](#Contents)
  - [Overview](#Overview)
  - [Dataset](#Dataset)
  - [Method](#Method)

## Overview
![MedKCoT](https://github.com/EnjoyFailure/MedKCoT/blob/main/framework.jpg)

We propose a framework of MedKCoT which mainly consists of three parts: (1) multi-modal knowledge retriever training, (2) multi-modal medical CoT generation and (3) MedVQA model training.

## Dataset
To evaluate the effectiveness of the proposed MedKCoT framework, we conduct experiments on two public Med-VOA datasets:(1)Slake and (2)VQA-RAD.
For Slake, we use its English version, which contains 642 radiology images and 7,033 question-answer pairs. For VQA-RAD, it contains 315 radiology images and 3,515 question–answer pairs.

## Method

### Step 0 
### Prepare environment
The environment for retriver training and CoT generation in requirements.txt
```
>>> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
the environment for MedVQA model training is based on LlamaFactory.

### Step 1
### retriver training
You can run the following script to train the retriver.
```
>>> cd MMKG_Retriver/scoring
>>> python iter_check_and_score.py # LVLM preference scoring
>>> cd MMKG_Retriver/training
>>> python main.py
```

### Step 2 
### CoT generation
You can run the following script to get the CoT from GPT-4.
```
>>> cd generation_CoT
>>> python gpt_generate_reason_slake_KG.py
```

### Step 3
### Train MedVQA model
Please use the obtained CoT and the original QA pair as training data and refer to the Settings of llamafactory to train the model.
https://github.com/hiyouga/LLaMA-Factory


