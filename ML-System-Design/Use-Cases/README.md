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

**Metrics Design**

Offline Metrics
- Use precision, recall, ranking loss, and logloss.

Online metrics
- Use A/B testing to compare Click Through Rates, watch time, and Conversion rates.

Retraining
- Train many times during the day to capture temporal changes.

Inference
- The latency needs to be under 200ms, ideally sub 100ms.


**Candidate Generation and Ranking Model**

![image](https://user-images.githubusercontent.com/46979228/182057235-27a2a9ba-b868-4233-ad36-9296429bf691.png)


Candidate generation model
- Matrix factorization
- Collaborative filtering
- In practice, for large scale system (Facebook, Google), we don’t use Collaborative Filtering and prefer low latency method to get candidate. One example is to leverage Inverted Index (commonly used in Lucene, Elastic Search). Another powerful technique can be found FAISS or Google ScaNN.

Ranking Model

**Calculation & estimation**

High-level System Design

![image](https://user-images.githubusercontent.com/46979228/182058109-2d23e3d2-94ca-4c9f-803b-f7e35057ab65.png)

**Scale the Design**
- In practice, we can also use Kube-proxy so the Candidate Generation Service can call Ranking Service directly, reducing latency even further.
