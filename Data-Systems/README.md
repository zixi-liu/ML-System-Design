
## Designing Data-Intensitive Applications

- [1. Reliable, Scalable, and Maintainable Applications](#1-Reliable,-Scalable,-and-Maintainable-Applications) 

### 1 Reliable, Scalable, and Maintainable Applications

#### Commonly needed functionality

- Databases
- Caches
- Search indexes
- Stream processing
- Batch processing

**Reliability**
- Hardware Faults
- Software Errors
- Human Errors
  - Decouple the places where people make the most mistakes from the places where they can cause failures.

**Scalibility**
- Describing Load
  - Load parameters - The best choice of parameters depends on the architecture of your system: it may be requests per second to a web
server, the ratio of reads to writes in a database, the number of simultaneously active users in a chat room, the hit rate on a cache, or something else. Perhaps the average case is what matters for you, or perhaps your bottleneck is dominated by a small number of extreme cases.
- Describing Performance
  - The response time is what the client sees: besides the actual time to process the request (the service time), it includes network delays and queueing delays. 
  - Latency is the duration that a request is waiting to be handled—during which it is latent, awaiting service。
- Approaches for Coping with Load
  - Scaling up (vertical scaling, moving to a more powerful machine).
  - Scaling out (horizontal scaling, distributing the load across multiple smaller machines. Distributing load across multiple machines is also
known as a shared-nothing architecture.)
  - Some systems are elastic, meaning that they can automatically add computing resources when they detect a load increase, whereas other systems are scaled manually.

**Maintainability**
