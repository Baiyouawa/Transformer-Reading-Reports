# Transformer-Reading-Reports🎉
# Transformer的论文阅读报告与代码理解🎉
## 🎄本项目包含对多篇Transformer论文及其衍生文章的论文精读报告及对代码的理解

## 文档结构：
```
Transformer-Reading-Report/
|—— Attention is all you need/ |—— Attention is all you need report.md #阅读报告
                               |—— images
                               |—— demo.py

|—— Bert/ |—— Bert report.md #阅读报告
          |—— images
          |—— demo.py

|—— VIT/  |—— VIT report.md #阅读报告
          |—— images
          |—— demo.py

|—— swintransformer |—— swintransformer.report # 阅读报告
                    |—— images
                    |—— demo.py

|—— GNN |—— GNN&Gragh.report #阅读报告
        |—— images

|—— Multimodal |—— Multimodal.report #阅读报告
               |—— demo.py

|—— conclude（PPT） |—— PPT #补充PPT文件

|—— README.md   #简介与链接
```
## 各个分项目重点内容：
### 包含链接与核心概念图，具体段落与补充知识请见论文阅读报告，实现复现后续补充
-----
### （一）：Attention is all you need
#### 原论文链接：[Attention is all you need](https://arxiv.org/abs/1706.03762)
#### 借鉴视频讲解：[跟李沐学AI：attention](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.999.0.0&vd_source=6e22f74cbbb0cdf9444235d6ad11aabf)
#### 论文阅读报告：[Attention is all you need report](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/Attention%20is%20all%20you%20need/Attention%20is%20all%20you%20need%20report.md)
#### demo代码：[transformer](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/Attention%20is%20all%20you%20need/demo.py)
**在过去的主流序列传导模型，主要是基于CNN或RNN开展的。**

**在RNN中（LSTM；GRU）等被认为是先进的方法，但由于其具有时序性，这种固有的顺序性限制了并行化，当输入的序列较长的时候，在计算效率和内存上存在很大限制，即便我们通过因子化技巧和条件计算等进行优化。**

**在使用CNN作为基础构建块的模型中，当输入与输出位置距离增加，所需要的操作数就随着位置之间距离增加而增长，也就导致学习远距离之间的依赖关系更加困难。**

**因此提出了完全基于注意力的transformer模型**

#### 核心图：
<table>
  <tr>
    <td><img src="images/Attention1.png" alt="Attention1" width="1300"/></td>
    <td><img src="images/Attention2.png" alt="Attention2" width="1300"/></td>
  </tr>
</table>

**左图是transformer的模型框架，右图为注意力机制**

**阶段一：**

**编码器**：首先输入进入Embedding嵌入层，将原始的输入转化为维度d为512的向量表示，这里的embedding内的权重×根号d，embedding学习后的向量维度大但是值小，需要扩大使得与位置编码相加两者等权。

随后对生成的向量与位置编码直接相加（因为我们输出的信息是value的加权和，并不具有时序性），这样使得向量信息既包含内容信息，也包含位置信息。

**解码器：** 解码器的输出再次作为输入进入解码器嵌入层进行类似的操作。

**阶段二**

**编码器：** 我们的输入序列经过三个线性变换形成QKV，随后通过线性层分解成多头进行点积注意力计算，多头输出后进行拼接形成最终输出结果（此时相当于做了一次汇聚）使得词语包含了我们想要的依赖关系的信息。随后通过残差连接进入layerNorm（层归一化）；

**解码器：** 类似的进行残差连接和多头注意力计算，这里存在掩码（目的是再训练时避免使用未来信息，所以通过设置后续非常小的value，使得我们的softmax函数后权重 为0）

#####  阶段三：

**编码器：** 经过层归一化后的输出通过残差连接进入MLP，再次实现将信息投影到我们更希望的语义空间中。再次通过归一化后作为Q，K进入解码器。

**解码器：** 上一步解码器输出作为V，与Q，K一起进行多头注意力计算，层归一化和线性层后实现输出，经过多层编码器-解码器架构后得到最终输出。

-----
###  (二）：Bert
#### 原论文链接：[Bert](https://arxiv.org/abs/1810.04805)
#### 借鉴视频链接：[李沐学AI：bert](https://www.bilibili.com/video/BV1PL411M7eQ?vd_source=88664659bdda4409e78f614f5f213ce8)
#### 论文阅读报告：[论文阅读报告](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/Bert/Bert%20report.md)
#### demo文件： [bert](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/Bert/demo.py)

**Bert是一种用于语言理解预训练的双向的transformer模型，他可以联系左右上下文，不同于Elmo，无需再对特定任务时进行任务框架的大量修改，而是联系上下文的微调方式，进而提升了在词级，句子级理解。**
#### 核心图：
<table>
  <tr>
    <td><img src="images/Bert1.png" alt="Attention1" width="1300"/></td>
    <td><img src="images/Bert2.png" alt="Attention2" width="1300"/></td>
  </tr>
</table>

 ##### 首先在右图我们可以看到输入操作：
##### 在Bert当中我们的输入token序列，可以是单个句子也可以是两个句子打包在一起。每个序列的第一个token是一个特殊的分类【CLS】，然后我们通过两种方式区别句子，首先我们用一个特殊的token【SEP】将他们分开，其次我们为每个token添加一个学习到的嵌入，来表示是句子AorB。也就是说，输入是由相应的token+段落嵌入+位置嵌入求和构建的。
  ##### 然后在左图我们可以看到模型架构：
##### Bert的模型核心就是基于原始实现的多层双向Transformer**编码器**，具体细节在上面可以看到。


-----
### (三）：VIT
#### 原论文链接：[VIT](https://arxiv.org/pdf/2010.11929)
#### 借鉴视频链接：[李沐学AI](https://www.bilibili.com/video/BV15P4y137jb/?spm_id_from=333.999.0.0&vd_source=6e22f74cbbb0cdf9444235d6ad11aabf)
#### 论文阅读报告：[论文阅读报告](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/VIT/VIT%20report.md)
#### demo文件：[VIT](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/VIT/demo.py)

##### 在NLP领域中transformer架构已经作为实际标准，但是在CV领域有限，尽管在前人实验中，有将注意力机制与卷积神经网络结合使用的，要么就是用于替换CNN中的某些组件。但是在VIT中，我们相当于直接将transformer进行应用于图像，拆分成补丁。但由于transformer框架没有过多的归纳偏置（在MLP上有），所以在中小型数据集上效果并不如ResNet，但是在大型数据集上有更加优秀的效果。并且随着模型参数增长没有出现过拟合的现象。

#### 核心图：
 <td><img src="images/VIT1.png" alt="VIT1" width="800"/></td>

**首先观察输入：**
输入为一个图片（224×224×3）（长宽RGB），我们将它打成16×16的patch，token就是16×16×3的维度（共计196个像素块）经过线性投影层做矩阵乘法得196×768+1×768（class）然后与位置编码通过sum方式进行更新。

**进入编码器：**
进入多头自注意力，拆分成KQV，同时又因为是多头自注意力；KQV变成了197×64，再经过拼接变为768维度，经过layernorm后进入MLP放大四倍，再投影回去。

最后：
- 全局信息聚合: 当输入序列（包括class token和patch tokens）通过多层Transformer编码器时，class token在每一层中与其他patch tokens交互并更新其表示。这种交互使得class token逐渐累积和整合整个图像的全局信息。

- 分类任务的特征图: 经过所有编码器层的处理后，class token最终包含了整个图像的全局特征。因为分类任务需要的是全局信息，class token成为了最合适的输出作为最终的分类特征图。

因此，class token 从输入阶段一直存在于所有编码器层，并在最后一层将其作为最终输出用于分类任务的决策。这意味着在每一层，class token都在继续累积来自其他patch tokens的信息，最终在输出时聚合了整个输入序列的全局信息。
-----

### （四）：Swintransformer：
#### 原论文链接：[Swin Transformer](https://arxiv.org/pdf/2103.14030)
#### 借鉴视频链接：[李沐学AI](https://www.bilibili.com/video/BV13L4y1475U?vd_source=88664659bdda4409e78f614f5f213ce8)
#### 论文阅读报告：[论文阅读报告](https://github.com/Baiyouawa/Transformer-Reading-Report/blob/main/swintransformer/SwinTransformer%20report.md)
#### demo文件：[Swin transformer](https://github.com/Baiyouawa/Transformer-Reading-Reports/blob/main/swintransformer/swin%20transformer.py)

Swin transformer就是让VIT也能像CNN一样分成很多block，形成层级式的提取，具有多尺度的概念。
通过移动窗口将自注意力计算限制在不重叠的局部窗口中，提高效率的同时，允许跨窗口通信连接。
这样使得再处理图像和视频时非常重要，在不同尺度（分辨率）上提取特征，捕捉图像中不同层次的细节和信息
**高分辨率：** 可以捕捉到图像的细节信息，比如纹理和边缘
**低分辨率：** 捕捉图像的整体结构和全局信息，进而提升鲁棒性。

#### 核心图：
<table>
  <tr>
    <td><img src="images/swin2.png" alt="swin2" width="1300"/></td>
    <td><img src="images/swin3.png" alt="swin3" width="1300"/></td>
  </tr>
</table>

#### 流程：
**首先我们的图片是224×224×3的维度，经过Patch partition（4×4）后，图片尺度变为56×56×48（48 = 4×4×3），接下来经过embedding层，我们设定超参数为C（96），这样经过全连接层，48变为96（56×56×96）；前面的参数拉直变为3136序列长度，96是token。**
**但是3136序列长度太长了，所以我们引入了窗口计算，每个窗口有7×7=49个patch；经过第一个部分Block后，得到56×56×96，随后进行Patch Merging以便获得多尺寸信息，类似于CNN池化。**
**随后将邻近的小Patch合成为一个大Patch，下采样两倍，所以隔一个点选一个，形成后为了将通道数仅仅扩大一倍，我们做一个1×1方向上卷积，通道数重新变为2，经过全局池化后得出总体。**

这种基于窗口的注意力计算相比于正常的计算减少了很大复杂度。

<table>
  <tr>
    <td><img src="images/swin4.png" alt="swin4" width="1300"/></td>
    <td><img src="images/swin5.png" alt="swin5" width="1300"/></td>
  </tr>
</table>

#### 窗口计算&掩码：
在经过移动窗口之后，会出现每个窗口Patch个数不同的情况，这也就导致我们不能转化为一个Batch进行计算，如果补充0（在外围）又会使得计算复杂度增加，所以我们采用掩码的方式（拼图）
**通过这种循环移位，分割填补。有的窗口应该进行注意力计算，同样，对于有的窗口中的其他部分，不应该进行窗口注意力计算，所以我们采取掩码方式，在矩阵乘法时，对于输出部分不需要的部分设定为较小的负数吗，进而在softmax操作之后，我们可以实现掩码。**

### （五）：GNN&Gragh：
#### 博客链接：[introduction](https://distill.pub/2021/gnn-intro/)
#### 原论文2链接：[GNN](https://arxiv.org/pdf/2310.11829)
#### 视频讲解链接：[李沐学AI](https://www.bilibili.com/video/BV1iT4y1d7zP?vd_source=88664659bdda4409e78f614f5f213ce8)
#### 论文阅读报告：[]()



#### 核心图：




