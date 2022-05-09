
# Designing Data-Intensive Applications

- [1. Reliable, Scalable, and Maintainable Applications](#1-Reliable-Scalable-and-Maintainable-Applications) 
- [2. Data Models and Query Languages](#2-Data-Models-and-Query-Languages) 
- [3. Storage and Retrieval](#3-Storage-and-Retrieval)

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

**Graph-Like Data Models**
- Vertices (also known as nodes or entities).
- Edges (also known as relationships or arcs).

Representing a property graph using a relational schema.
```
CREATE TABLE vertices (
vertex_id integer PRIMARY KEY,
properties json
);
CREATE TABLE edges (
edge_id integer PRIMARY KEY,
tail_vertex integer REFERENCES vertices (vertex_id),
head_vertex integer REFERENCES vertices (vertex_id),
label text,
properties json
);
CREATE INDEX edges_tails ON edges (tail_vertex);
CREATE INDEX edges_heads ON edges (head_vertex);
```

**Important aspects of graph model**
- Any vertex can have an edge connecting it with any other vertex. There is no schema that restricts which kinds of things can or cannot be associated.
- Given any vertex, you can efficiently find both its incoming and its outgoing edges, and thus traverse the graph—i.e., follow a path through a chain of vertices both forward and backward. 
- By using different labels for different kinds of relationships, you can store several different kinds of information in a single graph, while still maintaining a clean data model.

Graphs are good for *evolvability*: as you add features to your application, a graph can easily be extended to accommodate changes in your application’s data structures.
- Recursive common table expressions in SQL;

“NoSQL” datastores have diverged in two main directions:
- **Document databases** target use cases where data comes in self-contained documents and relationships between one document and another are rare.
- **Graph databases** go in the opposite direction, targeting use cases where anything is potentially related to everything.

One thing that document and graph databases have in common is that they typically don’t enforce a schema for the data they store, which can make it easier to adapt applications to changing requirements. However, your application most likely still assumes that data has a certain structure; it’s just a question of whether the schema is explicit (enforced on write) or implicit (handled on read).


### 3 Storage and Retrieval

Two families of storage engines: 
- Log-structured storage engines, 
- Page-oriented storage engines (such as B-trees).

**Hash Indexes**

**SSTables and LSM-Trees**
- Sorted String Table (the sequence of key-value pairs is sorted by key)
  - Compaction and merging process: merging several SSTable segments, retaining only the most recent value for each key.
  - Since read requests need to scan over several key-value pairs in the requested range anyway, it is possible to group those records into a block and compress it before writing it to disk. Each entry of the sparse in-memory index then points at the start of a compressed block. Besides saving disk space, compression also reduces the I/O bandwidth use.
- Log-Structured Merge-Tree

**B-Trees**

**Data Warehouse**

<img width="600" alt="image" src="https://user-images.githubusercontent.com/46979228/167333099-0d53461b-7ca1-4cb0-b2f2-e0e1068293ce.png">

- OLTP systems are usually expected to be highly available and to process transactions with low latency.

**Star schema**
- Fact table & dimension tables

**Column-Oriented Storage**
- Store all the values from each column together instead. If each column is stored in a separate file, a query only needs to read and parse those columns
that are used in that query, which can save a lot of work.
  - Parquet is a columnar storage format that supports a document data model, based on Google’s Dremel.

On a high level, storage engines fall into two broad categories: those optimized for transaction processing (OLTP), and those optimized for analytics (OLAP).

- OLTP systems are typically user-facing, which means that they may see a huge volume of requests. In order to handle the load, applications usually only touch a small number of records in each query. The application requests records using some kind of key, and the storage engine uses an index to find the data for the requested key. Disk seek time is often the bottleneck here.
- Data warehouses and similar analytic systems are less well known, because they are primarily used by business analysts, not by end users. They handle a much lower volume of queries than OLTP systems, but each query is typically very demanding, requiring many millions of records to be scanned in a short time. Disk bandwidth (not seek time) is often the bottleneck here, and columnoriented storage is an increasingly popular solution for this kind of workload.

On the OLTP side, we saw storage engines from two main schools of thought:
- The log-structured school, which only permits appending to files and deleting obsolete files, but never updates a file that has been written.
- The update-in-place school, which treats the disk as a set of fixed-size pages that can be overwritten （i.e. B-trees).

When queries require sequentially scanning across a large number of rows, indexes are much less relevant. Instead it becomes important to encode data very compactly, to minimize the amount of data that the query needs to read from disk (column-oriented storage helps achieve this goal).
