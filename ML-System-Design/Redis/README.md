# Redis

Redis is an in-memory data structure store, used as a distributed, in-memory key–value database, cache and message broker.


## Key value-oriented NoSQL

Key-value datastores can be understood as a big hash table. From a usage perspective, every value stored in the database has a key. The key can be used to search for values and the values can be deleted by deleting the key. Some popular choices in key-value databases are Redis, Riak, Amazon's DynamoDB, project voldermort, and more.

- **Insertion of data sets:** The insertions of data sets is very fast in key-value datastores and Redis is no exception.
- **Random reads:** Random reads are very fast in key-value datastores. In Redis, all the keys are stored in memory. This ensures faster lookups, so the read speeds are higher. While it will be great if all the keys and values are kept in memory, this has a drawback. The problem with this approach is that memory requirements will be very high. Redis takes care of this by introducing something called virtual memory. Virtual memory will keep all the keys in the memory but will write the least recently-used values to disk.
- **Fault tolerance:** Fault handling in Redis depends on the cluster's topology. Redis uses the master-slave topology for its cluster deployment. All the data in the master is asynchronously copied to the slave; so, in case the master node goes to the failure state, one of the slave nodes can be promoted to master using the Redis sentinel.
- **Eventual consistency:** Key-value datastores have master-slave topology, which means that once the master is updated, all the slave nodes are updated asynchronously. This can be envisaged in Redis since slaves are used by clients for a read-only mode; it is possible that the master might have the latest value written but while reading from the slave, the client might get the stale value because the master has not updated the slaves. Thus, this lag can cause inconsistency for a brief moment.
- **Load balancing:** Redis has a simple way of achieving load balancing. As previously discussed, the master is used to write data, and slaves are used to read the data. So, the clients should have the logic built into them, have the read request evenly spread across the slave nodes, or use third-party proxies, such as Twemproxy to do so.
- **Sharding:** It is possible to have datasets that are bigger than the available memory, which makes presharding the data across various peer nodes a horizontal scalable option.
- **Multi-data center support:** Redis and key-value NoSQL do not provide inherent multi-data center support where the replications are consistent. However, we can have the master node in one data center and slaves in the other data center, but we will have to live with eventual consistency.
- **Scalability:** When it comes to scaling and data partitioning, the Redis server lacks the logic to do so. Primarily, the logic to partition the data across many nodes should reside with the client or should use third-party proxies such as Twemproxy.
- **Manageability:** Redis as a key value NoSQL is simple to manage.
- **Client:** There are clients for Redis in Java, Python, and Node.js that implement the REdis Serialization Protocol (RESP).
