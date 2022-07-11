## 设计模式 Design Patterns

[**课程资源-设计模式之美**](https://time.geekbang.org/column/intro/250?tab=catalog)


**面向对象**
- 主流的编程范式：面向过程，面向对象和函数式编程。

面向对象编程的知识点：
- 面向对象的四大特性：封装、抽象、继承、多态
- 面向对象编程与面向过程编程的区别和联系
- 面向对象分析、面向对象设计、面向对象编程
- 接口和抽象类的区别以及各自应用场景
- 基于接口而非实现编程的设计思想
- 多用组合少用继承的设计思想
- 面向过程的贫血模型和面向对象的充血模型

### 01 封装、抽象、继承、多态

**封装 Encapsulation**

封装也叫做信息隐藏或数据访问保护。对于封装特性，我们需要访问权限控制支持(i.e. private, public, etc.)。

**抽象 Abstraction**

抽象讲如何隐藏方法的具体实现，让使用者只需要关心方法提供了哪些功能，不需要知道这些功能是如何实现的。

**继承 Inheritance**

继承的好处是代码复用。Java extends, Python (), ...

**多态 Polymorphism**

多态指子类可以替换父类。只要两个类具有相同的方法，就可以实现多态，并不要求两个类之间有任何关系。

*utils 类： 设计时最好细化成FileUtils， IOUtils，StringUtils， UrlUtils...

### 02 接口vs抽象类的区别

## 设计原则

**SOLID原则：**
- 单一职责原则
- 开闭原则
- 里式替换原则
- 接口隔离原则
- 依赖反转原则

### 单一职责原则 SRP

一个类或者模块只负责完成一个指责（或者功能）。模块可以看成比类更加粗粒度的代码块，模块中包含多个类，多个类组成一个模块。

举例：设计两个粒度细的类：订单类和用户类。

### 开闭原则 OCP

软件实体（模块、类、方法）应该对扩展开放（新增模块、类、方法等），对修改关闭（修改模块、类、方法等）。

重构
- 将入参封装成类；
- 引入Handler概念；

- *思考需求变更，如何设计代码结构，事先留好扩展点。然后将可变部分封装起来，隔离变化，提供抽象化的不可变接口，给上层系统使用。*

[[The Open-Closed Principle, Robert C. Martin 2006]](https://courses.cs.duke.edu/fall07/cps108/papers/ocp.pdf)

### 里氏替换原则 LSP

子类对象(object of subtype/derived class)能够替换程序中的父类对象(object of base/parent class)出现的任何地方，并且保证原来程序的逻辑行为不变及正确性不被破坏。

子类可以改变函数的内部实现逻辑，但不能改变函数原有的行为约定。

### 接口隔离原则 ISP

客户端不应该被强迫依赖它不需要的接口。

### 依赖注入 DI

**控制反转 IOC**

依赖注入：不通过new()的方式在类内部创建依赖类对象，而是将依赖的类对象在外部创建好之后，通过构造函数、函数参数等方式传递给类使用。


**KISS原则， YAGNI原则， DRY原则**

**高内聚，低耦合**
- 高内聚：相近的功能应该放到同一个类中；
- 低耦合：类与类之间的依赖关系简单清晰；

## 规范与重构

- 单元测试 Unit Testing
- 集成测试 Integration Testing



