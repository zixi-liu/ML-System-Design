
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
  - [一文看懂Attention](https://zhuanlan.zhihu.com/p/91839581)
  - [遍地开花的 Attention](https://zhuanlan.zhihu.com/p/77307258)
    - 计算区域
      - Soft Attention：对所有key求权重概率，每个key都有一个对应的权重，是一种全局的计算方式。
      - Hard Attention：精准定位到某个key，相当于这个key的概率是1，其余key的概率全部是0。
      - Local Attention：对一个窗口区域进行计算。先用Hard方式定位到某个地方，以这个点为中心可以得到一个窗口区域，在这个小区域内用Soft方式来算Attention。
    - 结构层次
      - 单层Attention：用一个query对一段原文进行一次attention。
      - 多层Attention：一般用于文本具有层次关系的模型。
      - 多头Attention：用到了多个query对一段原文进行了多次attention，每个query都关注到原文的不同部分，相当于重复做多次单层attention。
    - self attention: Q(Query), K(Key), V(Value)三个矩阵均来自同一输入.
    - multi-head self-attention: 词的representation有侧重点的包含其他词汇信息。
    - cross-attention: 查询来自一个输入序列，而键和值来自另一个输入序列。
- encoder中的attention都是self-attention，decoder则除了self-attention还有cross-attention。
- [理解Attention:从起源到MHA,MQA和GQA](https://www.linsight.cn/3dc22f96.html#mha)
- Transformer模型架构中的其他部分
  - Layer Normalization: 加速模型的优化速度。
  - Feed Forward Network: FFN的加入引入了非线性(ReLu激活函数)，变换了attention output的空间, 从而增加了模型的表现能力。
  - Positional Encoding
    
### The encoder stack

Each layer contains two main sublayers: a multi-headed attention mechanism and a fully connected position-wise feedforward network.

Notice that a residual connection surrounds each main sublayer, sublayer(x), in the Transformer model. These connections transport the unprocessed input x of a sublayer to a layer normalization function. This way, we are certain that key information such as positional encoding is not lost on the way. 
- LayerNormalization (x + Sublayer(x))

#### Input embedding

#### Positional encoding

Use a unit sphere to represent positional encoding with sine and cosine values that will thus remain small but useful.

- [Transformer 中的 positional embedding](https://zhuanlan.zhihu.com/p/359366717)
- [Attention is all you need; Attentional Neural Network Models](https://www.youtube.com/watch?v=rBCqOTEfxvg)
  - make a query with a vector and look at similar things in the past
  - multi-modal
- [碎碎念：Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628)

#### Adding positional encoding to the embedding vector

### The decoder stack

At a given position, the following words are **masked** so that the Transformer bases its assumptions on its inferences without seeing the rest of the sequence.

## 3. Fine-Tuning BERT Models

