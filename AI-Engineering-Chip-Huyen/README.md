## Chapter 1. Introduction to Building AI Applications with Foundation Models

**model as a service**

**From Language Models to Large Language Models**
- key is self-supervision

**two main types of language models**
- masked language models
  - trained to predict missing tokens anywhere in a sequence, using the context from both before and after the missing tokens.
  - commonly used for non-generative tasks such as sentiment analysis and text classification. 
  - they are also useful for tasks requiring an understanding of the overall context, such as code debugging, where a model needs to understand both the preceding and following code to identify errors.
- autoregressive language models
  - trained to predict the next token in a sequence, using only the preceding tokens.
  - can continually generate one token after another.
  - models of choice for text generation.
 
![image](https://github.com/user-attachments/assets/8dace71b-522d-4268-aaa4-34eab6787441)

**Self-supervision**
- instead of requiring explicit labels, the model can infer labels from the input data.

**From Large Language Models to Foundation Models**
- CLIP: instead of manually generating labels for each image, they found (image, text) pairs that co-occurred on the internet.
  - CLIP is an embedding model, trained to produce joint embeddings of both texts and images.
  - Multimodal embedding models like CLIP are the backbones of generative multimodal models, such as Flamingo, LLaVA, and Gemini (previously Bard).

**Get the model to generate what you want**
- prompt engineering
- retrieval-augmented generation (RAG)
- finetune

**Three Layers of the AI Stack**
- Application development
- Model development
  - frameworks for modeling, training, finetuning, and inference optimization.
- Infrastructure
  - tooling for model serving, managing data and compute, and monitoring.
 
**model adaptation**
- Prompt-based techniques, which include prompt engineering, adapt a model without updating the model weights.
- Finetuning, on the other hand, requires updating model weights.
- different training phases
  - Pre-training
  - Finetuning
  - Post-training
    - goal of post-training is to align the model with human preferences.
- Dataset engineering
  - expertise in data is useful when examining a model, as its training data gives important clues about that model’s strengths and weaknesses.
- Inference optimization
  - quantization, distillation, and parallelism
- Evaluation


## Chapter 2. Understanding Foundation Models

**Training Data**

**Modeling**
- Transformer architecture
  - two problems with seq2seq:
    - the vanilla seq2seq decoder generates output tokens using only the final hidden state of the input.
    - the RNN encoder and decoder mean that both input processing and output generation are done sequentially, making it slow for long sequences.
  - transformer architecture addresses both problems with the attention mechanism
  - Inference for transformer-based language models
    - Prefill
    - Decode
- Attention mechanism
  - query vector (Q) represents the current state of the decoder at each decoding step.
  - key vector (K) represents a previous token. If each previous token is a page in the book, each key vector is like the page number.
  - value vector (V) represents the actual value of a previous token, as learned by the model.
  - computes how much attention to give an input token by performing a dot product between the query vector and its key vector.
  - <img width="657" alt="image" src="https://github.com/user-attachments/assets/ac5d76fe-9167-4c8d-91a2-9045c3ac7011" />
  - multi-headed
    - allow the model to attend to different groups of previous tokens simultaneously.
- Transformer block
  - An embedding module before the transformer blocks
    - consists of the embedding matrix and the positional embedding matrix
  - An output layer after the transformer blocks
  - <img width="659" alt="image" src="https://github.com/user-attachments/assets/98ac412f-da93-41fd-8ad4-a52c00453d62" />
    - Larger dimension values result in larger model sizes.
    - Increased context length impacts the model’s memory footprint, it doesn’t impact the model’s total number of parameters.
- Other model architectures
   - Modeling long sequences remains a core challenge in developing LLMs.
   - mixture-of-experts (MoE)
     - An MoE model is divided into different groups of parameters, and each group is an expert. Only a subset of the experts is active for (used to) process each token.
   - When discussing model size, it’s important to consider the size of the data it was trained on.
   - A more standardized unit for a model’s compute requirement is FLOP, or floating point operation. 

**Post-Training**
- a pre-trained model typically has two issues:
  - self-supervision optimizes the model for text completion, not conversations.
  - if the model is pre-trained on data indiscriminately scraped from the internet, its outputs can be racist, sexist, rude, or just wrong.
- in general, post-training consists of two steps:
  - Supervised finetuning (SFT): Finetune the pre-trained model on high-quality instruction data to optimize models for conversations instead of completion.
  - Preference finetuning: Further finetune the model to output responses that align with human preference. Preference finetuning is typically done with reinforcement learning (RL).
- Post-training, in general, optimizes the model to generate responses that users prefer.
- <img width="649" alt="image" src="https://github.com/user-attachments/assets/99f53dc7-8b0e-452d-bbed-aeefc2abdcf1" />
  - a combination of pre-training, SFT, and preference finetuning is the popular solution for building foundation models today.
- Supervised Finetuning
- Preference Finetuning
  - RLHF consists of two parts:
    - Train a reward model that scores the foundation model’s outputs.
    - Optimize the foundation model to generate responses for which the reward model will give maximal scores.
  - Reward model
    - Given a pair of (prompt, response), the reward model outputs a score for how good the response is.
    - Finetuning using the reward model
      - proximal policy optimization (PPO)

**Sampling**
- Sampling Fundamentals
  - greedy sampling: picking the most likely outcome
    - for a language model, greedy sampling creates boring outputs.
  - Sampling Strategies
    - Temperature
      - a higher temperature reduces the probabilities of common tokens, and as a result, increases the probabilities of rarer tokens. This enables models to create more creative responses.
    - Top-k
      - A smaller k value makes the text more predictable but less interesting, as the model is limited to a smaller set of likely words.
    - Top-p
      - the model sums the probabilities of the most likely next values in descending order and stops when the sum reaches p.
    - Stopping condition
      - ask models to stop generating after a fixed number of tokens.
  - Test Time Compute
    - instead of generating only one response per query, you generate multiple responses to increase the chance of good responses.
    - use beam search to generate a fixed number of most promising candidates (the beam) at each step of sequence generation.
    - increase the diversity of the outputs, because a more diverse set of options is more likely to yield better candidates.
  
  **Structured Outputs**

**The Probabilistic Nature of AI**
- Inconsistency is when a model generates very different responses for the same or slightly different prompts.
  - Same input, different outputs
    - cache the answer so that the next time the same question is asked, the same answer is returned.
    - fix the model’s sampling variables, such as temperature, top-p, and top-k values.
    - fix the seed variable, which you can think of as the starting point for the random number generator used for sampling the next token.
  - Slightly different input, drastically different outputs
    - get models to generate responses closer to what you want with carefully crafted prompts and a memory system.
- Hallucination is when a model gives a response that isn’t grounded in facts. Currently two hypotheses about why language models hallucinate:
  - 1. a language model hallucinates because it can’t differentiate between the data it’s given and the data it generates.
    - Self-delusion: Starting with a generated sequence slightly out of the ordinary, the model can expand upon it and generate outrageously wrong facts.
    - Snowballing hallucinations: After making an incorrect assumption, a model can continue hallucinating to justify the initial wrong assumption.
    - hallucinations can be mitigated by two techniques.
      - reinforcement learning, in which the model is made to differentiate between user-provided prompts (called observations about the world in reinforcement learning) and tokens generated by the model (called the model’s actions).
      - supervised learning, in which factual and counterfactual signals are included in the training data.
  - 2. hallucination is caused by the mismatch between the model’s internal knowledge and the labeler’s internal knowledge.
    - During SFT, models are trained to mimic responses written by labelers. If these responses use the knowledge that the labelers have but the model doesn’t have, we’re effectively teaching the model to hallucinate.

## Chapter 3. Evaluation Methodology

**Understanding Language Modeling Metrics**
- Cross entropy, perplexity, BPC, and BPB
  - cross entropy measures how difficult it is for a model to predict the next token.
  - perplexity measures the amount of uncertainty it has when predicting the next token.
    - More structured data gives lower expected perplexity.
    - The bigger the vocabulary, the higher the perplexity.
    - The longer the context length, the lower the perplexity. (The more context a model has, the less uncertainty it will have in predicting the next token. )
    - For a given model, perplexity is the lowest for texts that the model has seen and memorized during training.
      - useful for detecting data contamination—if a model’s perplexity on a benchmark’s data is low, this benchmark was likely included in the model’s training data.
