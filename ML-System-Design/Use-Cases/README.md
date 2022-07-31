## Primer Knowledge

**Feature Engineering**

- Feature Hashing
- Crossed Features
- Embeddings
  - [DoorDash - Store2Vec](https://doordash.news/company/personalized-store-feed-with-vector-embeddings/)

**Training Pipeline**
- Data partitioning
  - parquet format
- Handle imbalance class distribution
  - Use class weights in loss function
  - Naive resampling
  - Synthetic resampling
- Retraining requirements
  - A common design pattern is to use a scheduler to retrain models on a regular basis.
  
**Inference**
- Imbalance workload
  - One common pattern is to split workloads onto multiple inference servers. (similar to load balancer)
  - Aggregator Service can pick workers through one of the following ways:
  - a) Work load
  - b) Round Robin
  - c) Request parameter
- Serving logics and multiple models
- Non-stationary problem
  - Data distribution shift 
- [EE-trade off] Exploration vs. exploitation: Thompson Sampling

**Metrics Evaluation**
- Offline metrics (i.e. logloss etc.)
- Online metrics (i.e. Lift in revenue or click through rate)

![image](https://user-images.githubusercontent.com/46979228/182039940-d33de79d-2f7d-466e-8705-7564208642fc.png)

## Video Recommendation

**Problem Statement**
- Maximize users’ engagement and recommend new types of content to users.
