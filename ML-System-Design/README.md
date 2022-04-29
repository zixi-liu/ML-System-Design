## ML System Fundamentals

### Framing ML Problems

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


### Data Systems

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

### Training Data



