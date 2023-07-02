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

  
 

