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
