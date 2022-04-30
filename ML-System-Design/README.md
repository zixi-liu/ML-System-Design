## ML System Fundamentals

- [1. Framing ML Problems](#1-Framing-ML-Problems) 
- [2. Data Systems](#2-Data-Systems)
- [3. Training Data](#3-Training-Data)
- [4. Feature Engineering](#4-Feature-Engineering)


### 1 Framing ML Problems

**Use Case to Decouple Objectives**

Imagine you’re building a system to rank items on users’ newsfeed. Your goal is to maximize users’ engagement while minimizing the spread of extreme views and misinformation.

- Filter out spam
- Filter out NSFW content
- Filter out misinformation
- Rank posts by quality
- Rank posts by engagement: how likely users will click on it

Essentially, you want to minimize *quality_loss*: the difference between each post’s predicted quality and its true quality. Similarly, to rank posts by engagement, you first need to predict the number of clicks each post will get. You want to minimize engagement_loss: the difference between each post’s predicted clicks and its actual number of clicks.

One approach is to combine these two losses into one loss and train one model to minimize that loss.

loss = 𝛼 quality_loss + 𝛽 engagement_loss

Another approach is to train two different models, each optimizing one loss. So you have two models:
- quality_model minimizes quality_loss and outputs the predicted quality of each post.
- engagement_model minimizes engagement_loss and outputs the predicted number of clicks of each post.

You can combine the outputs of these two models and rank posts by their combined scores:

𝛼 quality_score + 𝛽 engagement_score


### 2 Data Systems

**Data Sources**

- User input data
- System-generated data
- Internal DBs
- Third party data

**Data Models**

- Relational Models
- NoSQL Models
  - Document Model
  - Graph Model

**Data Storage Engines and Processing**

Two types of workloads that databases are optimized for: 
- Transactional processing
  - A transaction refers to any kind of actions that happen online: tweeting, ordering a ride through ridesharing service, uploading a new model, watching a YouTube video, etc.
  - OnLine Transaction Processing (OLTP) - need to be processed fast (low latency).
- Analytical processing
  - OnLine Analytical Processing (OLAP) - requires aggregating data in columns across multiple rows of data.

**Stream Processing**

### 3 Training Data

#### Sampling

**Non-Probability Sampling**

When the selection of data isn't based on any probability criteria.
- Convenience sampling: sample data based on availability.
- Snowball sampling: future samples are selected based on existing samples.
- Judgment sampling: expert decide what samples to include.
- Quota sampling: select samples based on quotas for certain slices of data without any randomization.

**Simple Random Sampling**

**Stratified Sampling**

**Weighted Sampling**

**Importance Sampling**

**Reservoir Sampling**

Useful when dealing with incoming stream of data.

#### Labeling

**Hand Labeling**

- Label Multiplicity
- Data Lineage

**Handling the Lack of Hand Labels**

- Weak Supervision
  - Labeling Function (LF) can encode heuristics: keyword heuristic, regular expressions, database lookup, outputs of other models etc.
- [[Semi-supervision]](https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf)
  - Assumes that data samples that share similar characteristics share the same labels. 
- Transfer Learning
  - A model developed for a task is reused as the starting point for a model on a second task.
- Active Learning
  - The most straightforward heuristic is uncertainty measurement — label the examples that your model is the least certain about hoping that they will help your model learn the decision boundary better. 
  - Another common heuristic is based on disagreement among multiple candidate models.

#### Class Imbalance

CLass imbalance make learning difficult for the following reasons:
- Insufficient signal for your model to learn to detect the minority classes.
- Get stuck in a non-optimal solution by learning a simple heuristic instead of learning useful about the underlying structure of the data.
- Aymmetric costs of error - the cost of wrong prediction on an example of the rare class might be much higher than a wrong prediction on an example of the majority class.

Classical examples of tasks with class imbalance: fraud detection, churn prediction, disease screening, resume screening etc.

**Handling Class Imbalance**

- Using the right evaluation metrics
- Resampling data
- Algorithm-level methods
  - Cost-sensitive learning: individual loss function is modified to take into account this varying cost.
  - Class-balanced loss: make the weight of each class inversely proportional to the number of samples in that class, so that the rarer classes have higher weights.
  - Focal loss: adjust the loss so that if a sample has a lower probability of being right, it’ll have a higher weight.

### 4 Feature Engineering

#### Data Augmentation

- Simple Label-Preserving Transformations 
- Perturbation
  - Adding noisy samples to our training data can help our models recognize the weak spots in their learned decision boundary and improve their performance. 
  - Adversarial augmentation is less common in NLP, but perturbation has been used to make models more robust. One of the most notable examples is BERT.
- Data Synthesis
  - Used templates to bootstrap training data.

#### Learned Features vs. Engineered Features

- For domain-specific tasks such as predicting whether a transaction is fraudulent, you might need subject matter expertise with banking and frauds to be able to extract useful features.

#### Common Feature Engineering Operations

**Handling Missing Values**
- Deletion
- Imputation

**Scaling**

**Discretization**

**Encoding Categorical Features**
- Hashing

**Feature Crossing**
- Combine two or more features to generate new features.
  - DeepFM and xDeepFM are the family of models that have successfully leverage explicit feature interactions for recommendation systems and click-through-rate prediction tasks.

**Discrete and Continuous Positional Embeddings**



