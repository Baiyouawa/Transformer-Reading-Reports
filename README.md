# Reading-Reports-and-codes🎉
### 🎄本项目包含对本人相关领域的论文合集与代码复现，用于了解领域，发现问题，学习最新的模型方法。
### 文档结构：
```
Reading-Reports-and-codes/
|—— GPT-ST/ |—— Group-reporting.pptx #阅读报告   （时空预测领域）🌠
|—— 阅读方法.md  #对阅读论文方法的总结
|—— README.md   #简介与链接

```

### 论文1:GPT-ST:时空图神经网络的生成式预训练：🌠
论文链接：[GPT-ST](https://arxiv.org/abs/2311.04245v1)

**核心思想：设计一种能够无缝衔接到下游任务的现有时空预测模型的生成式预训练**

作者灵感来源于迅速发展的端到端模型，尝试通过设计全新的预训练方式来提升下游任务性能，作者对应相对的问题进行了算法模块的设计，结果在性能和效率上都有很大的提升，但是模型仍然存在效率和泛化性问题，后续作者团队进行相关展望。

**（1）：特定时空模式的定制表示 --->  定制的参数学习器**

**（2）：不同层次上时空依赖性 --->   分层空间编码器**

**（3）：进一步提高模型在聚类间学习能力和模型推理能力  ---> 自适应掩码策略**



