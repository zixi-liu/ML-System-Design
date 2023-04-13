## 大数据经典论文解读

**大数据的核心概念**
- 能够伸缩到一千台服务器以上的分布式数据处理集群的技术。
- 这个上千个节点的集群，是采用廉价的 PC 架构搭建起来的。
- “把数据中心当作是一台计算机”（Datacenter as a Computer）。

GFS、MapReduce 和 Bigtable 完成了“存储”“计算”“实时服务”这三个核心架构的设计。

Spark 通过把数据放在内存而不是硬盘里（数据缓存在内存），大大提升了分布式数据计算性能。

**实时数据处理的抽象进化**

![image](https://user-images.githubusercontent.com/46979228/204672310-98595d7e-236b-4684-8b6e-18d3bb864248.png)


**RoadMap**

分布式系统
- 可靠性 - 数据复制 （主从架构，多主架构，无主架构）
- 可扩展性 - 数据分区 （区间分区，一致性Hash分区）
- 可维护性 - 容错与恢复 

存储引擎
- 事务：预写日志（WAL）、快照（Snapshot）和检查点（Checkpoints）以及写时复制（Copy-on-Write）
- 写入与存储
- 数据的序列化

计算引擎
- 批式处理
- 流式处理
- 以批代流
- 数据传输

## 大数据处理

#### MapReduce

<img width="581" alt="image" src="https://user-images.githubusercontent.com/46979228/231879985-7130f5c4-e59e-44b3-8560-8f47a774b455.png">

2008 年以后，Google 改进了 MapReduce 的分片功能，引进了动态分片技术 (dynamic sharding），大大简化了使用者对于分片的手工调整。

自动优化的需求：
- 对数据处理进行高度抽象，去除冗余步骤。
- 计算资源的自动弹性分配。

数据处理的描述语言，与背后的运行引擎解耦合开来。

统一批处理和流处理的编程模型。

<img width="585" alt="image" src="https://user-images.githubusercontent.com/46979228/231882728-a56d5fa6-6979-44fd-9b48-789c31484dda.png">

#### Top K算法当数据规模变大会遇到哪些问题

- 内存占用。
- 磁盘 I/O 等延时问题。

#### 分布式系统

Best Practices
- 99.9% Availability是什么
  - SLA（Service-Level Agreement）服务等级协议。最常见的四个 SLA 指标，可用性、准确性、系统容量和延迟。
  - **可用性（Availabilty）**： 可用性指的是系统服务能正常运行所占的时间百分比。
  - **准确性（Accuracy）**
