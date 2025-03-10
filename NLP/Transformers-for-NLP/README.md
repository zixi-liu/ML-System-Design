
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

- Creating sentences, label lists, and adding BERT tokens
- Processing the data
  - sets the maximum length of a sequence to 128, and the sequences are padded
- Creating attention masks
- Splitting the data into training and validation sets
- BERT model configuration

[详解 Bert：Bidirectional Encoder Representations from Transformers](https://zhuanlan.zhihu.com/p/521330227)
- 模型输入/输出
  - token embedding：可以随机初始化，也可以通过word2ve、glove等算法进行预训练的初始化。
  - position embedding：transformer通过余弦位置编码，bert通过学习位置embedding。
  - segment embedding：有部分NLP任务的输入不是单个句子，而是句子对（比如：句子匹配），因此需要segment embedding来对句子的序号进行区分。
  - 最后由三个embedding相加得到word embedding。
  - 每个序列前面增加一个特殊的类别标记[CLS]。
  - 在句子A和句子B之间插入[SEP],用于分隔开来两个句子。
  - 输出分为pooler output和sequence output。
- 模型结构
  - bert_base: 12层tansformer encoder, 隐层大小768，self-attention的head数12，总参数110M
  - bert_large: 24层tansformer encoder, 隐层大小1024，self-attention的head数16，总参数340M
- 模型训练
  - 两个自监督任务
    - Masked Language Model（MLM）
      - 随机mask掉15%的词（字），然后通过非监督学习的方法来进行预测。
    - Next Sentence Prediction（NSP）
      - NSP是选择一个句子对（A，B），其中B有50%是A的下一句，有50%是随机从语料库中挑选的，让模型预测是否B为A的下一句。
  - 微调
    - bert之后接全连接+softmax进行分类。

[Bert系列文章]
- [Bert系列一：词表示，从one-hot到transformer](https://zhuanlan.zhihu.com/p/365774595)
  - one-hot
  - Word2vec: CBOW, Skip-gram
  - ELMo
  - Transformer
- [Bert系列四：生成模型 GPT 1.0 2.0 3.0](https://zhuanlan.zhihu.com/p/365554706)

[有监督微调]
- [微调基本概念](https://github.com/wdndev/llm_interview_note/blob/main/05.%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83/1.%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5/1.%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5.md)
  - Parameter-Efficient Fine-Tuning（PEFT）：通过冻结预训练模型的某些层，并仅微调特定于下游任务的最后几层来实现这种效率。
  - 高效微调技术可以粗略分为以下三大类
    - 增加额外参数（A）
      - 类适配器（Adapter-like）
      - 软提示（Soft prompts）
    - 选取一部分参数更新（S）
    - 引入重参数化（R）
  - 多种不同的高效微调方法对比
    - 选择性层调整（Selective Layer Tuning）：可以只微调层的一个子集，而不是微调模型的所有层。这减少了需要更新的参数数量。
    - 适配器（Adapters）：适配器层是插入预训练模型层之间的小型神经网络。在微调过程中，只训练这些适配器层，保持预先训练的参数冻结。通过这种方式，适配器学习将预先训练的模型提取的特征适应新任务。
    - 稀疏微调（Sparse Fine-Tuning）：传统的微调会略微调整所有参数，但稀疏微调只涉及更改模型参数的一个子集。这通常是基于一些标准来完成的，这些标准标识了与新任务最相关的参数。
    - 低秩近似（Low-Rank Approximations）：另一种策略是用一个参数较少但在任务中表现相似的模型来近似微调后的模型。
    - 正则化技术（Regularization Techniques）：可以将正则化项添加到损失函数中，以阻止参数发生较大变化，从而以更“参数高效”的方式有效地微调模型。
    - 任务特定的头（Task-specific Heads）：有时，在预先训练的模型架构中添加一个任务特定的层或“头”，只对这个头进行微调，从而减少需要学习的参数数量。

## 4. Pre-training a RoBERTa Model from Scratch

## 5. Downstream NLP Tasks with Transformers

**Transformer Performance versus Human Baselines**
- Evaluating Models with Metrics
  - Accuracy
  - F1
  - MCC
  - Benchmark Tasks and Datasets
    - GLUE - SuperGLUE

## 6. Machine Translation with the Transformer

## 7. The Rise of Suprahuman Transformers with GPT-3 Engines


**Other Resources**

- Hard Attention vs Soft Attention
- [LLM的3种架构：Encoder-only、Decoder-only、encode-decode](https://zhuanlan.zhihu.com/p/642923989)
  - encoder-only类型的更擅长做分类；包括情感分析，命名实体识别.
  - encoder-decoder类型的擅长输出强烈依赖输入的，比如翻译和文本总结，而其他类型的就用decoder-only，如各种Q&A。
  - 虽然encoder-only没有decoder-only类型的流行，但也经常用于模型预训练.
- [Transformer模型及其变种（BERT、GPT）](https://zhuanlan.zhihu.com/p/706094599)
