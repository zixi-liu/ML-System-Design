
## For my dartmouth lab

- [Masked Autoencoder](https://arxiv.org/pdf/2111.06377)
- [Masked Autoencoders论文阅读](https://zhuanlan.zhihu.com/p/432978632)
  - 语言具有高度的语义信息特征，而视觉图像具有高度冗余的特点.
  - mask大部分随机patch来降低冗余信息.
  - MAE通过预测每个mask patch的像素值来重建输入.
  - 解码器输出中的每个元素都是代表一个patch的像素值向量.
  - 解码器的最后一层是一个线性投影，其输出的通道数等于一个patch中像素值的数量.
  - 解码器的输出reshape以后形成重构图像。计算重建图像和原始图像之间的均方差损失（MSE）
