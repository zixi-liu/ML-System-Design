
## Tutorials

[**Data Attribution at Scale**](https://icml.cc/virtual/2024/tutorial/35228)
- A data attribution method seeks to connect model behavior at test time with data.
- Corroborative data attribution: When can I trust my model's output?
- Game-theoretic data attribution: What training data is to blame for my model's error?
   - Goal is assign "fair credit" to different sources for the outcome
- Predictive data attribution: How robust is my modeling to malicious training data?
- Model debugging with data attribution
   - Common approach: give each training sample an "importance", then inspect
   - What do these "importances" yield?
      - Most "positive importance" samples
      - Most "negative importance" samples
