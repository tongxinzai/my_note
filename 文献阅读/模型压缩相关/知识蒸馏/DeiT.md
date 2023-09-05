- **Training data-efficient image transformers & distillation through attention.** *Hugo Touvron, M. Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Herv'e J'egou.* **International Conference on Machine Learning, 2020** [(PDF)](../../Notetool/papers/Training%20data-efficient%20image%20transformers%20&%20distillation%20through%20attention.pdf data-efficient image transformers & distillation through attention.pdf>)  [(arxiv)](https://arxiv.org/abs/2012.12877)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fad7ddcc14984caae308c397f1a589aae75d4ab71%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/ad7ddcc14984caae308c397f1a589aae75d4ab71)
#### DeiT：注意力也能蒸馏

ViT 在大数据集 ImageNet-21k（14million）或者 JFT-300M（300million） 上进行训练，Batch Size 128 下 NVIDIA A100 32G GPU 的计算资源加持下预训练 ViT-Base/32 需要3天时间。

Facebook 与索邦大学 Matthieu Cord 教授合作发表 Training data-efficient image transformers（DeiT） & distillation through attention，DeiT 模型（8600万参数）仅用一台 GPU 服务器在 53 hours train，20 hours finetune，仅使用 ImageNet 就达到了 84.2 top-1 准确性，而无需使用任何外部数据进行训练。性能与最先进的卷积神经网络（CNN）可以抗衡。所以呢，很有必要讲讲这个 DeiT 网络模型的相关内容。

下面来简单总结 DeiT：

> DeiT 是一个全 Transformer 的架构。其核心是提出了针对 ViT 的教师-学生蒸馏训练策略，并提出了 token-based distillation 方法，使得 Transformer 在视觉领域训练得又快又好。

#### DeiT 相关背景

ViT 文中表示数据量不足会导致 ViT 效果变差。针对以上问题，DeiT 核心共享是使用了蒸馏策略，能够仅使用 ImageNet-1K 数据集就就可以达到 83.1% 的 Top1。

那么文章主要贡献可以总结为三点：

1. 仅使用 Transformer，不引入 Conv 的情况下也能达到 SOTA 效果。
2. 提出了基于 token 蒸馏的策略，针对 Transformer 蒸馏方法超越传统蒸馏方法。
3. DeiT 发现使用 Convnet 作为教师网络能够比使用 Transformer 架构效果更好。

正式了解 DeiT 算法之前呢，有几个问题需要去了解的：ViT的缺点和局限性，为什么训练ViT要准备这么多数据，就不能简单快速训练一个模型出来吗？另外 Transformer 视觉模型又怎么玩蒸馏呢？

#### ViT 的缺点和局限性

Transformer的输入是一个序列（Sequence），ViT 所采用的思路是把图像分块（patches），然后把每一块视为一个向量（vector），所有的向量并在一起就成为了一个序列（Sequence），ViT 使用的数据集包括了一个巨大的包含了 300 million images的 JFT-300，这个数据集是私有的，即外部研究者无法复现实验。而且在ViT的实验中作者明确地提到：

> "That transformers do not generalize well when trained on insufficient amounts of data."
   当不使用 JFT-300 大数据集时，效果不如CNN模型。也就反映出Transformer结构若想取得理想的性能和泛化能力就需要这样大的数据集。DeiT 作者通过所提出的蒸馏的训练方案，只在 Imagenet 上进行训练，就产生了一个有竞争力的无卷积 Transformer。

#### DeiT 具体方法

为什么DeiT能在大幅减少 **1\. 训练所需的数据集** 和 **2\. 训练时长** 的情况下依旧能够取得很不错的性能呢？我们可以把这个原因归结为DeiT的训练策略。ViT 在小数据集上的性能不如使用CNN网络 EfficientNet，但是跟ViT结构相同，仅仅是使用更好的训练策略的DeiT比ViT的性能已经有了很大的提升，在此基础上，再加上蒸馏 (distillation) 操作，性能超过了 EfficientNet。

假设有一个性能很好的分类器作为teacher model，通过引入了一个 Distillation Token，然后在 self-attention layers 中跟 class token，patch token 在 Transformer 结构中不断学习。

Class token的目标是跟真实的label一致，而Distillation Token是要跟teacher model预测的label一致。
![](图片/知识蒸馏/deit/deit1.png)
对比 ViT 的输出是一个 softmax，它代表着预测结果属于各个类别的概率的分布。ViT的做法是直接将 softmax 与 GT label取 CE Loss。

$CE Loss(x, y) = - \sum y_i * log(x_i)$

而在 DeiT 中，除了 CE Loss 以外，还要 1）定义蒸馏损失；2）加上 Distillation Token。

- **定义蒸馏损失**

蒸馏分两种，一种是软蒸馏（soft distillation），另一种是硬蒸馏（hard distillation）。软蒸馏如下式所示，$Z_s$ 和 $Z_t$ 分别是 student model 和 teacher model 的输出，KL 表示 KL 散度，psi 表示softmax函数，lambda 和 tau 是超参数：
$$\mathcal{L}_{\text {global }}=(1-\lambda) \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{\mathrm{s}}\right), y\right)+\lambda \tau^{2} \mathrm{KL}\left(\psi\left(Z_{\mathrm{s}} / \tau\right), \psi\left(Z_{\mathrm{t}} / \tau\right)\right)$$
硬蒸馏如下式所示，其中$\mathcal{L}_{CE}$ 表示交叉熵：
$$\mathcal{L}_{\text {global }}^{\text {hardDistill }}=\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y\right)+\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y_{\mathrm{t}}\right)$$
学生网络的输出 $Z_s$ 与真实标签之间计算 CE Loss 。如果是硬蒸馏，就再与教师网络的标签取 CE Loss。如果是软蒸馏，就再与教师网络的 softmax 输出结果取 KL散度 。

值得注意的是，Hard Label 也可以通过标签平滑技术 （Label smoothing） 转换成Soft Labe，其中真值对应的标签被认为具有 1- esilon 的概率，剩余的 esilon 由剩余的类别共享。

- **加入 Distillation Token**

Distillation Token 和 ViT 中的 class token 一起加入 Transformer 中，和class token 一样通过 self-attention 与其它的 embedding 一起计算，并且在最后一层之后由网络输出。

而 Distillation Token 对应的这个输出的目标函数就是蒸馏损失。Distillation Token 允许模型从教师网络的输出中学习，就像在常规的蒸馏中一样，同时也作为一种对class token的补充。
#### DeiT 具体实验

实验参数的设置：图中表示不同大小的 DeiT 结构的超参数设置，最大的结构是 DeiT-B，与 ViT-B 结构是相同，唯一不同的是 embedding 的 hidden dimension 和 head 数量。作者保持了每个head的隐变量维度为64，throughput是一个衡量DeiT模型处理图片速度的变量，代表每秒能够处理图片的数目。
![](图片/知识蒸馏/deit/deit3.png)
![](图片/知识蒸馏/deit/deit2.png)

作者首先观察到使用 CNN 作为 teacher 比 transformer作为 teacher 的性能更优。上图中对比了 teacher 网络使用 DeiT-B 和几个 CNN 模型 RegNetY 时，得到的 student 网络的预训练性能以及 finetune 之后的性能。
其中，DeiT-B 384 代表使用分辨率为 384×384 的图像 finetune 得到的模型，最后的那个小蒸馏符号 alembic sign 代表蒸馏以后得到的模型。

1. **蒸馏方法对比**

下图是不同蒸馏策略的性能对比，label 代表有监督学习，前3行分别是不使用蒸馏，使用soft蒸馏和使用hard蒸馏的性能对比。前3行不使用 Distillation Token 进行训练，只是相当于在原来 ViT 的基础上给损失函数加上了蒸馏部分。![](图片/知识蒸馏/deit/deit4.png)

![](图片/知识蒸馏/deit/deit3.png)
对于Transformer来讲，硬蒸馏的性能明显优于软蒸馏，即使只使用 class token，不使用 distill token，硬蒸馏达到 83.0%，而软蒸馏的精度为 81.8%。

随着微调，class token 和 Distillation Token 之间的相关性略有增加。除此之外，蒸馏模型在 accuracy 和 throughput 之间的 trade-off 甚至优于 teacher 模型，这也反映了蒸馏的有趣之处。

- **对比实验**

作者还做了一些关于数据增强方法和优化器的对比实验。Transformer的训练需要大量的数据，想要在不太大的数据集上取得好性能，就需要大量的数据增强，以实现data-efficient training。几乎所有评测过的数据增强的方法都能提升性能。对于优化器来说，AdamW比SGD性能更好。
![](图片/知识蒸馏/deit/deit4.png)

此外，发现Transformer对优化器的超参数很敏感，试了多组 lr 和 weight+decay。stochastic depth有利于收敛。Mixup 和 CutMix 都能提高性能。Exp.+Moving+Avg. 表示参数平滑后的模型，对性能提升只是略有帮助。最后就是 Repeated augmentation 的数据增强方式对于性能提升帮助很大。
#### 小结

DeiT 模型（8600万参数）仅用一台 GPU 服务器在 53 hours train，20 hours finetune，仅使用 ImageNet 就达到了 84.2 top-1 准确性，而无需使用任何外部数据进行训练，性能与最先进的卷积神经网络（CNN）可以抗衡。其核心是提出了针对 ViT 的教师-学生蒸馏训练策略，并提出了 token-based distillation 方法，使得 Transformer 在视觉领域训练得又快又好。