
## Chapter 1. Computer Abstractions and Technology

### Understanding Program Performance

The performance of a program depends on a combination of the effectiveness of the algorithms used in the program, the software systems used to create and translate the program into machine instructions, and the effectiveness of the computer in executing those instructions, which may include input/output (I/O) operations.

<img width="706" alt="image" src="https://github.com/zixi-liu/ML-System-Design/assets/46979228/96dbbf52-7d14-4cb6-81d4-b1b91f75824e">

#### Below your program

Compiler: A program that translates high-level language statements into assembly language statements.

Assembler: A program that translates a symbolic version of instructions into the binary version.

## Chapter 6. Parallel Processors From Client to Cloud

### SIMD: Single Intruction Multiple Data

For example, a single SIMD instruction might add 64 numbers by sending 64 data streams to 64 ALUs to form 64 sums within a single clock cycle.

SIMD works best when dealing with arrays in for loops. Hence, for parallelism
to work in SIMD, there must be a great deal of identically structured data, which
is called *data-level parallelism*. SIMD is at its weakest in case or switch
statements, where each execution unit must perform a diff erent operation on its
data, depending on what data it has

### Vector Architecture

Th e basic philosophy of vector architecture is to collect data elements from memory, put them in order into a large set of registers, operate on them sequentially in registers using pipelined execution units, and then write
the results back to memory. A key feature of vector architectures is then a set of
vector registers. Th us, a vector architecture might have 32 vector registers, each
with 64 64-bit elements.

#### Resources:
- [Computer Organization and Design](https://www.cse.iitd.ac.in/~rijurekha/col216/edition5.pdf)
- [Introduction to GPU Architecture](http://www.haifux.org/lectures/267/Introduction-to-GPUs.pdf)

##### Throughput Processing
1. Understand space of GPU core (and throughput CPU core) designs

[Cuda基础-03.访存](https://zhuanlan.zhihu.com/p/565199964)
- Register 寄存器是 GPU 上最快的存储器，因此使用它们来增加数据重用是一个重要的性能优化。
- 寄存器不是永久的，因此存储在寄存器中的数据只有在线程执行期间才可用。

3. Optimize shaders/compute kernels
4. Establish intuition: what workloads might benefit from the design of
these architectures?
