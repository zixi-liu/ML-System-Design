## A Survey of Large Language Models

[A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223)

#### 1. Introduction

**Task solving capacity of LLMs**

- Statistical LM: Specific task helper
- Neural LM: Task-agnostic feature learner (Word2vec etc.)
- Transferable NLP task solver - Pre-trained LM (ELMO, BERT, GPT-1/2)
- General-purpose task solver - LLM (GPT-3/4, ChatGPT, Claude)

**Three major differences between LLMs and PLMs**

- LLMs display some surprising emergent abilities that may not be observed in previous smaller PLMs.
- Unlike small PLMs, the major approach to accessing LLMs is through the prompting interface (e.g., GPT-4 API).
- The training of LLMs requires extensive practical experiences in large-scale data processing and distributed parallel training.

**Recent advances in LLMs from four major aspects**

- Pre-training (how to pretrain a capable LLM)
- Adaptation (how to effectively adapt pre-trained LLMs for better use)
- Utilization (how to use LLMs for solving various downstream tasks)
- Capability evaluation (how to evaluate the abilities of LLMs and existing empirical findings). 

#### 2. Overview

Large language models (LLMs) refer to Transformer language models that contain hundreds of billions (or more) of parameters
- GPT-3
- PaLM
- Galactica
- LLaMA

**Formulation of Scaling Laws for LLMs**
- KM scaling law
  - model the power-law relationship of model performance with respective to three major factors, namely model size (N), dataset size (D), and the amount of training compute (C)

**Emergent Abilities of LLMs**
- "the abilities that are not present in small models but arise in large models"
- three typical emergent abilities for LLMs
  - In-context learning
  - Instruction following
  - Step-by-step reasoning (chain-of-thought (CoT) prompting strategy)
 
**Key Techniques for LLMs**
