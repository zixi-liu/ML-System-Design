
# Designing Data-Intensive Applications

- [1. Reliable, Scalable, and Maintainable Applications](#1-Reliable-Scalable-and-Maintainable-Applications) 
- [2. Data Models and Query Languages](#2-Data-Models-and-Query-Languages) 

### 1 Reliable Scalable and Maintainable Applications

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


### 2 Data Models and Query Languages

**Relational Model Versus Document Model**
- NoSQL
  - The JSON representation has better locality than the multi-table schema. The one-to-many relationships from the user profile to the user’s positions, educational history, and contact information imply a tree structure in the data, and the JSON representation makes this tree structure explicit.

**Many-to-One and Many-to-Many Relationships**
- If the data in your application has a document-like structure (i.e., a tree of one-tomany relationships, where typically the entire tree is loaded at once), then it’s probably a good idea to use a document model. The relational technique of shredding—splitting a document-like structure into multiple tables can lead to cumbersome schemas and unnecessarily complicated application code.

- **schema-on-read** (the structure of the data is implicit, and only interpreted when the data is read)
  - The schema-on-read approach is advantageous if the items in the collection don’t all have the same structure for some reason (i.e., the data is heterogeneous).
- **schema-on-write** (the traditional approach of relational databases, where the schema is explicit and the database ensures all written data conforms
to it)

**Data locality for queries**
- If your application often needs to access the entire document (for example, to render it on a web page), there is a performance advantage to this storage locality.

**Query Languages for Data**
- SQL as a declarative query language
- MapReduce Querying based on the map (also known as collect) and reduce (also known as fold or inject) functions that exist in many functional programming languages.
  - MapReduce is a fairly low-level programming model for distributed execution on a cluster of machines.
  - MongoDB 2.2 added support for a declarative query language called the aggregation pipeline.
