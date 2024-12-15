
[Transformers for Natural Language Processing](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1803247339)


## 1. What are Transformers?

**Foundation Models**
- Homogenization: use one model to perform a wide variety of tasks.
- The present ecosystem of transformer models
  - Model architecture
  - Data
  - Computing power
  - Prompt engineering
- Each word (token) of a sequence is related to all the other words of a sequence.

## 2. Getting Started with the Architecture of the Transformer Model

![image](https://github.com/user-attachments/assets/2cc2c926-aeaa-4c2e-b619-2ab7a127750f)

[解读Transformer模型和Attention机制](https://zhuanlan.zhihu.com/p/104393915)
- **核心思想**：使用attention机制, 在一个序列的不同位置之间建立distance = 1的平行关系，从而解决RNN的长路径依赖问题(distance = N)。
- Simple RNN: encoder output会作为decoder的initial states的输入。随着decoder长度的增加，encoder output的信息会衰减。
- Contextualized RNN
- Contextualized RNN with soft align (Attention)
  - "attention"操作的目的就是计算当前token与每个position之间的"相关度"，从而决定每个position的vector在最终该timestep的context中占的比重有多少。最终的context即encoder output每个位置vector表达的加权平均。
  - [transformer中的Q,K,V](https://www.zhihu.com/question/427629601)
    - query对应的是需要被表达的序列(称为序列A)，key和value对应的是用来表达A的序列(称为序列B)。
    - multi-head self-attention: 词的representation有侧重点的包含其他词汇信息。
    - encoder-decoder attention: 接下一个词时有focus的与源语言进行对比。
    - self-attention: encoder中的self-attention的query, key, value都对应了源端序列(即A和B是同一序列)，decoder中的self-attention的query, key, value都对应了目标端序列。
    - cross-attention: 



### The encoder stack
