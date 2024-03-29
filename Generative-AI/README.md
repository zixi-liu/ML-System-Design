### Week 1

#### Text Generation Before Transformers

- RNNs (models need to see more than the previous few words)

#### Transformer Architecture

- Self-attention
- Multi-headed Self-attention

**Architecture**
- Tokenization
- Pass to Embedding layer
- Positional Encoding
  - Position embeddings + Token embeddings

- **Encoder** inputs prompts with contextual understanding and produces one vector per input token.
- **Decoder** accepts input tokens and generates new tokens.

![image](https://github.com/zixi-liu/ML-System-Design/assets/46979228/0acaa3f7-6bba-42fb-9bd1-2245d455b01a)

#### Prompting and prompt engineering

- In-context Learning (ICL) - zero-shot inference, one-shot, few shot

#### Generative configuration - inference parameters

- Max new tokens
- Sample top K
- Sample top P
- Temperature

Greedy vs. random sampling 
- Greedy: the word/token with highest probability is selected
- Random(-weighted) sampling: select a token using a random-weighted strategy across the probablities of all tokens. (reduce the probably that words are repeated)

Top-k and top-p sampling
- Top-k: select an output from the top-k results after applying random-weighted strategy using the probabilities
- Top-p: select an output using the random-weighted strategy with top-ranked consecutive results by probability and with a cumulative probability <= p.

Temperature
- Control the randomness - shape of probablity distribution
  - The higher the temperature, the higher the randomness.
  - Cooler temperature (<1) indicates strongly peaked probability distribution. Higher temperature (>1) indicates a broader, flatter probability distribution.

#### Project Lifecycle

![image](https://github.com/zixi-liu/ML-System-Design/assets/46979228/b75140c7-991d-4c73-9be4-c2aae4d1b72f)

#### Model architectures and pre-training objectives

**Autoencoding models**
- Masked Language Modeling (MLM)
  - Objective: Reconstruct text (denoising) with bidirectional context
  - Use cases: sentiment analysis, named entity recognition, word classification

**Autoregressive models**
- Causal Language Modeling (CLM)
  - Objective: Predict next token with undirectional context
  - Use cases: text generation, other emergent behavior (depends on model size)
 
**Sequence-to-sequence models**
- Span corruption (sentinel token)
- Reconstruct span
  - Use cases: transalation, text summarization, question answering

#### Computational Challenges

- 1B parameters = 4 x 10^9 bytes = 4GB

Reducing Memmory
- Quantization
  - BFLOAT16

**Efficient multi-GPU compute strategies**

- Distributed Data Parallel (DDP)
- Fully Sharded Data Paralel (FSDP)
  - ZeRO (zero data overlap across GPUs, sharding parameters, gradients, and optimizer states

**Scaling Laws**
- Scaling choice
  - Increase dataset size (number of tokens)
  - Increase model size (number of parameters)

1 "petaflop/s-day" = # floating point operations performed at rate of 1 petaFLOP per second for one day

Very large models maybe over-parameterized and under-trained.

**Pre-training for domain adaption**

i.e. legal language, medical language, etc.

BloombergGPT - a LLM for Finance

### Week 2

#### Instruction fine-tuning

Fine-tuning is a supervised process using task-specific examples.
- Prompt-completion pairs

#### Fine-tuning on a single task

**Catastrophic forgetting**
- Fine-tuning can significantly increase the performance of a model on a specific task, but can lead to reduction in ability on other tasks.

**How to avoid catastrophic forgetting**
- Fine-tune on multiple tasks at the same time
- Consider Parameter Efficient Fine-tuning (PEFT) - greater robustness

**Multi-task, instruction fine-tuning**
- Instruction fine-tuning with FLAN
  - FLAN models refer to a specific set of instructions used to perform instruction fine-tuning.

#### Model Evaluation

LLM Evaluation
- Accuracy
- ROUGE
  - Used for text summarization
  - Compares a summary to one or more reference summaries
  - ROUGE-1 Recall = unigram matches / unigrams in reference
  - ROUGE-1 Precision = unigram matches / unigrams in output
  - ROUGE-1 F1 = 2 precision x recall / precision + recall
  - ROUGE-2 (using bigrams)
  - ROUGE-L (longest common subsequence)
- BLEU Score
  - Used for text translation
  - Compared to human-generated translations
  - Avg(precision across range of n-gram sizes)

#### Benchmarks

#### Parameter efficient fine-tuning (PEFT)

Selective
- Select subset of initial LLM parameters to fine-tune

Reparameterization
- Reparameterize model weights using a low-rank representation (LoRA - Low-Rank Adpaptation of LLMs)

Additive 
- Add trainable layers or parameters to model (Adapters, Soft Prompts)

#### LoRA

- Freeze most of the original LLM weights.
- Inject 2 rank decomposition matrices.
- Train the weights of the smaller matrices.

[**LoRA: Low-Rank Adaptation of Large Language Models 简读**](https://zhuanlan.zhihu.com/p/514033873)

[**论文阅读：LORA-大型语言模型的低秩适应**](https://zhuanlan.zhihu.com/p/611557340)

#### Soft Prompts

- Prompt tuning
  - Weights of model frozen and soft prompt trained
  - Switch out soft prompt at inference time to change task

### Week 3

#### Reinforcement Learning from Human Feedback (RLHF)

**Reinforcement Learning**

Maximize reward received for actions
- Agent
- Environment

![image](https://github.com/zixi-liu/ML-System-Design/assets/46979228/aae96108-8029-41b1-9652-b8ad1c94bbf7)

#### Obtaining feedback from humans

- Definen your model alignment criterion
- Obtain human feedback through labeler workforce

Prepare labeled data for training
- Convert rankings into pairwise training data for the reward model

#### Reward Model

#### Fine-tuning with reinforcement learning

- Proximal Policy Optimization

#### RLHF: Reward Hacking

- Maintain a reference model and measure the KL Divergence Shift Penalty between reference model and RL-updated LLM.
  - KL divergence penalty gets added to reward
- Evaluate the human-aligned LLM
  - Evaluate using toxicity score

#### Scaling Human Feedback

**Constitutional AI**
- Supervised Learning Stage
  - Red Teaming
  - Response, critique and revision

#### Lab 
The fine-tuning loop consists of the following main steps:
- Get the query responses from the policy LLM (PEFT model).
- Get sentiments for query/responses from hate speech RoBERTa model.
- Optimize policy with PPO using the (query, response, reward) triplet.

#### Model Optimization for Deployment

**Distillation**
- Teach model -> student model by using a distillation loss; Temperature;

**Quantization**
- Post-Training Quantization (PTQ)
  - Reduce precision of model weights

**Pruning**
- Remove model weights with values close or equal to zero

#### Generative AI Project Lifecycle Cheat Sheet

![image](https://github.com/zixi-liu/ML-System-Design/assets/46979228/b35792cb-ab6e-4c02-b629-0ea3b5875cb8)


