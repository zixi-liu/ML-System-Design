## A Survey of Large Language Models

[A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223)

### 1. Introduction

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

### 2. Overview

#### 2.1 Background for LLMs

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
- Scaling.
- Training.
  - To support distributed training, several optimization frameworks have been released to facilitate the implementation and deployment of parallel algorithms,
such as DeepSpeed and Megatron-LM.
  - Also, optimization tricks are also important for training stability and model performance, e.g., restart to overcome training loss spike and mixed precision training.
- Ability eliciting.
  - As the technical approach, it is useful to design suitable task instructions or specific in-context learning strategies to elicit such abilities. For instance, chain-of-thought prompting has been shown to be useful to solve complex reasoning tasks by including intermediate reasoning steps.
- Alignment tuning.
  - It is necessary to align LLMs with human values, e.g., helpful, honest, and harmless.
  - reinforcement learning with human feedback
- Tools manipulation.

#### 2.2 Technical Evolution of GPT-series Models

The basic principle underlying GPT models is to compress the world knowledge into the decoder-only
Transformer model by language modeling, such that it can recover (or memorize) the semantics of world knowledge and serve as a general-purpose task solver. Two key points to the success are **(I) training decoder-only Transformer language models that can accurately predict the next word** and **(II) scaling up the size of language models**.
- GPT-1[2017]: unsupervised pre-training and supervised fine-tuning
- GPT-2: unsupervised language modeling, without explicit fine-tuning using labeled data.
  - a probabilistic form for multi-task solving, i.e., p(output|input, task)
  - A basic understanding of this claim is that each (NLP) task can be considered as the word prediction problem based on a subset of the world text.
- GPT-3[2020]: in-context learning (ICL), which utilizes LLMs in a fewshot or zero-shot way.
  - pre-training predicts the following text sequence conditioned on the context, while ICL predicts the correct task solution, which can be also formatted as a text sequence, given the task description and demonstrations.
  - **Capacity Enhancement**
    - Training on code data
    - Human alignment
- The Milestones of Language Models.
  - ChatGPT[2022]
  - GPT-4[2023]: extended the text input to multimodal signals.
    - GPT-4 responds more safely to malicious or provocative queries, due to a sixmonth iterative alignment (with an additional safety reward signal in the RLHF training).[red teaming]
  - GPT-4V, GPT-4 turbo, and beyond.
 
 ### 3. Resources of LLMs

 ####  3.1 Publicly Available Model Checkpoints or APIs
 - LLaMA.
 - Mistral.
 - Gemma.
 - Qwen.
 - GLM.
 - Baichuan.

#### 3.2 Commonly Used Corpora for Pre-training

- Web pages
- Books & Academic Data
- Wikipedia
- Code
- Mixed Data

#### 3.3 Commonly Used Datasets for Fine-tuning

- instruction tuning (supervised fine-tuning)
- alignment tuning

### 4. Pre-Training

#### 4.1 Data Collection and Preparation

**4.1.1 Data Source**
- General Text Data.
- Specialized Text Data.

**4.1.2 Data Preprocessing**
- Filtering and Selection.
  - classifier-based: trains a selection classifier based on high-quality texts and leverages it to identify and filter out low-quality data.
  - heuristic-based
    - *Language based filtering:* If a LLM would be mainly used in the tasks of certain languages, the text in other languages can be filtered.
    - *Metric based filtering:* Evaluation metrics about the generated texts, e.g., perplexity, can be employed to detect and
remove unnatural sentences.
    - *Statistic based filtering:* Statistical features of a corpus, e.g., the punctuation distribution, symbol-to-word ratio, and sentence length, can be utilized to measure the text
quality and filter the low-quality data.
    - *Keyword based filtering:* Based on specific keyword set, the noisy or unuseful elements in the text, such as HTML tags, hyperlinks, boilerplates, and offensive words, can be identified and removed.
- De-duplication.
  - duplicate data in a corpus would reduce the diversity of language models, which may cause the training process to become unstable and thus affect the model performance.
  - de-duplication can be performed at different granularities, including sentence-level, document-level, and dataset-level de-duplication.
- Privacy Reduction.
- Tokenization.
  - Byte-Pair Encoding (BPE) tokenization.
  - WordPiece tokenization.
  - Unigram tokenization.

**4.1.3 Data Scheduling**

- the proportion of each data source (data mixture)
  - Increasing the diversity of data sources.
  - Optimizing data mixtures.
  - Specializing the targeted abilities.
- the order in which each data source is scheduled for training (data curriculum).


#### 4.2 Architecture

**4.2.1 Typical Architectures**
- Encoder-decoder Architecture
- Causal Decoder Architecture
  - Incorporates the unidirectional attention mask, to guarantee that each input token can only attend to the past tokens and itself.
- Prefix Decoder Architecture
  - Revises the masking mechanism of causal decoders, to enable performing bidirectional attention over the prefix tokens and unidirectional attention only on generated tokens. 
- Mixture-of-Experts

**4.2.2 Detailed Configuration**

Discussed four major parts of the Transformer, including normalization, position embeddings, activation functions, and attention and bias.

Normalization Methods
- LayerNorm
- RMSNorm
- DeepNorm

Activation Functions

Position Embeddings
- Absolute position embedding
- Relative position embedding

Attention
- Full attention
- Sparse attention
- Multi-query/grouped-query attention
- FlashAttention

**4.2.3 Pre-training Tasks**
- Language Modeling: LM task aims to autoregressively predict the target tokens xi based on the preceding tokens x<i in a sequence.
- Denoising Autoencoding: The language models are trained to recover the replaced tokens x˜.
- Mixture-of-Denoisers.

**4.2.4 Decoding Strategy**
- Improvement for Greedy Search.
  - Beam search:  retains the sentences with the n (beam size) highest probabilities at each step
during the decoding process, and finally selects the generated response with the top probability.
  - Length penalty: Since beam search favours shorter sentences.
- Improvement for Random Sampling
  - Temperature sampling
  - Top-k sampling
  - Top-p sampling

#### 4.3 Model Training

**4.3.1 Optimization Setting**
- Batch Training
- Learning Rate
- Optimizer
- Stabilizing the Training

### Post-Training of LLMs

#### 5.1 Instruction Tuning
- Key Factors for Instruction Dataset Construction
  - Scaling the instructions
  - Formatting design
  - Instruction quality improvement
  - Instruction selection

**5.1.2 Instruction Tuning Strategies**
- Balancing the Data Distribution
- Combining Instruction Tuning and Pre-Training
- Multi-stage Instruction Tuning
- Other Practical Tricks
  - Efficient training for multi-turn chat data
  - Establishing self-identification for LLM

#### 5.2 Alignment Tuning

### 6. Utilization
-  in-context learning
-  chain-of-thought prompting
-  planning: solving complex tasks, which first breaks them down into smaller sub-tasks and then generates a plan of action to solve these sub-tasks one by one.


