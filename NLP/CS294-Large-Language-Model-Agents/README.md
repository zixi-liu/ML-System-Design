## LLM Reasoning

- few-shot prompting
- adding reasoning process
- chain-of-throught (CoT) promting

Regardless of training, fine-tuning, or prompting, when provided with examples that include intermediate steps, LLMs will
respond with intermediate steps.

- Constant-depth transformers can solve any inherently serial problem as long as it generates sufficiently long intermediate reasoning steps.
- Transformers which directly generate final answers either requires a huge depth to solve or cannot solve at all.

**Key Takeaways**
- 1. Pre-trained LLMs, without further finetuning, has been ready for step-by-step reasoning, but we need a non-greedy decoding strategy to elicit it.
- 2. When a step-by-step reasoning path is present, LLMs have much higher confidence in decoding the final answer than direct-answer decoding.
 
## LLM Agents

