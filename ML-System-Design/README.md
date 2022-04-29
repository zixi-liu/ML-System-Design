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


