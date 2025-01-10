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


