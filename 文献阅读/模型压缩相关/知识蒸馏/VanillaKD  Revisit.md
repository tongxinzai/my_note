**VanillaKD  Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale.** *Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang.* **arXiv.org, 2023** [(PDF)](<../../NoteTool/papers/VanillaKD  Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale.pdf>)  [(arxiv)](https://arxiv.org/abs/2305.15781)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fda1946bb4220e743e8f46946397a9b31e609df74%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/da1946bb4220e743e8f46946397a9b31e609df74)

## Abstract
大规模数据集上训练的大型模型取得了巨大的成功，证明规模是实现优异结果的关键因素。因此，基于小规模数据集设计知识蒸馏（Knowledge Distillation，KD）方法的合理性已成为当务之急。本文中，我们发现了先前KD方法中存在的小数据陷阱，导致对大规模数据集（如ImageNet-1K）上的原始KD框架的能力低估。具体而言，我们展示了采用更强的数据增强技术和使用更大的数据集可以直接减小原始KD框架与其他精心设计的KD变种之间的差距。这凸显了在实际场景中设计和评估KD方法的必要性，摆脱小规模数据集的限制。我们对原始KD及其变种在更复杂方案中的研究包括更强的训练策略和不同的模型容量，表明原始KD在大规模场景中简洁而有效。不加花哨的设计，我们在ImageNet数据集上获得了最先进的ResNet50、ViT-S和ConvNeXtV2-T模型，分别达到了83.1%、84.3%和85.0%的top-1准确率。
## Vanilla KD can not achieve satisfactory results on small-scale dataset 
### 2.1 Review of knowledge distillation

基于先前训练的教师模型使用的信息来源，知识蒸馏（Knowledge Distillation，KD）技术可以大致分为两类：一种是利用教师模型的输出概率（基于logits）的方法，另一种是利用教师模型的中间表示（基于hint）的方法。基于logits的方法利用教师模型的输出作为辅助信号来训练一个较小的模型，被称为学生模型。
$$L _ { k d } = \alpha D _ { c l s } ( p ^ { s } , y ) + ( 1 - \alpha ) D _ { k d } ( p ^ { s } , p ^ { t } )\tag{1}$$
在这里，p^s 和 p^t 分别是学生模型和教师模型的 logits。y 是真实标签的独热编码(one-hot)。D_cls 和 D_kd 分别是分类损失和蒸馏损失，例如交叉熵和 KL 散度。
超参数 α 决定了两个损失项之间的平衡。为了方便起见，我们在接下来的所有实验中将 α 设为 0.5。
除了 logits，中间的表示(hint)（特征）也可以用于知识蒸馏。考虑到学生特征 F^s 和教师特征 F^t，基于提示的蒸馏可以通过以下方式实现：
$$L _ { h i n t } = D _ { h i n t } ( T _ { s } ( F ^ { s } ) , T _ { t } ( F ^ { t } ) )\tag{2}$$
在这里，$T_s$ 和 $T_t$ 是用于对齐两个特征的转换模块。$D_hint$ 是特征差异的度量，例如 $L_1$ 或 $L_2$ 范数。在常见的做法中，基于提示的损失通常与分类损失一起使用，并且本文在实验中遵循了这个设置。

### 2.2 Vanilla KD can not achieve satisfactory results on small-scale dataset!
![](图片/知识蒸馏/VanillaKD%20Revisit/tab1.png)
- **Impact of limited model capacity**
在表1a中，当使用更强的训练方法后，原始kd与DKD和DIST方法之间的差距惊人的缩小了。这说明原始KD方法的能力不足归咎于不足的训练，而不是其本身能力有限。
- **Impact of small dataset scale.**
在表1b中，作者在CIFAR-100上进行实验。此时原始kd与其余两种方法在不同的训练策略下依旧保持较大差距。并且在res56-res20上的进度差距更大了，因此说明原始kd不仅局限于不足的训练，还受到小数据集的影响。
- **Discussion**
在图一中可以看到随着数据集的不断增长，vanilla KD 与其他的方法在性能上的差距逐渐减小。
![](图片/知识蒸馏/VanillaKD%20Revisit/fig1.png)

## 3 Evaluate the power of vanilla KD on large-scale dataset

### 3.1 Experimental setup
- **dataset**：使用imagenet-1k、Imagenet-Real、Imagenet-V2数据集
-  **Model**：主要使用BEiTv-L  Res50 师生对，还使用其他ResNet网络和ConvNeXt作为教师模型，Vit 和ConvNeXtV2作为学生模型。
- **训练策略**：用了A1和A2两种策略，A1稍微比A2效果好（也没说具体是什么策略 ，引用了一下deit那篇文章，我有空回去把deit仔细看看）
- **baseline 蒸馏方法**:
	1. Logit-based：原始KD ,KD, DKD , and DIST ；
	2. hint-based ：CC , RKD , CRD , and ReviewKD .

### 3.2 Logits-based methods consistently outperform hint-based methods
![](图片/知识蒸馏/VanillaKD%20Revisit/tab3.png)
在表3中可以看到基于logit的蒸馏方法普遍比基于中间的表示(hint)的蒸馏方法性能更好。并且基于中间表示的方法需要更长的训练时间，突出了其效率和效能的局限性。

**Discussion**：
基于中间表示的方法大体上被基于logit的方法超过，作者推测原因是学生和教师模型在处理复杂分布时的能力不同，采用中间特征的表示方法会不合适。此外，在异构的教师和学生模型中，不同的学习表示会阻碍特征对齐过程。
为了对此进行分析，我们进行了中心核分析（CKA）[37]，以将Res50提取的特征与Res152和BEiTv2-L提取的特征进行比较。如图2所示。res152相比于BETiTv2-L更接近于res50。在基于hint的异构教师学生网络方法上除了性能上次优外，从CKA还表现其在异构场景下的不兼容。
![](图片/知识蒸馏/VanillaKD%20Revisit/fig2.png)

### 3.3 Vanilla KD preserves strength with increasing teacher capacity
**表3**还显示了使用更强的异构BEiTv2-L教师的原始KD和其他两种基于logits的baseline之间的比较。这位老师在开源模型中的ImageNet-1K验证中达到了SOTA的top-1精度。这些结果清楚地表明，在以前的研究中，原始KD被低估了，因为它们是在小规模数据集上设计和评估的。当与强大的同质和异质教师一起训练时，vanilla KD在保持其简单性的同时，实现了与最先进方法相当的竞争性能。

### 3.4 Robustness of vanilla KD across different training components（消融实验）![](图片/知识蒸馏/VanillaKD%20Revisit/tab4.png)
- **Ablation on Loss functions.**
损失函数的消融实验硬标签采用交叉熵损失(CE)和二进制交叉熵损失（BCE）；软标签采用KL散度和二进制KL散度。
KL散度的二进制版本：
$$L _ { B K L } = \sum _ { i \in y } [ - p _ { i } ^ { t } \log ( p _ { i } ^ { s } /p _ { i } ^ { t } ) - ( 1 - p ^ { t } ) \log ( ( 1 - p ^ { s } ) / ( 1 - p ^ { t } ) ) ]$$
结果如**表4**所示，使用BKL会产生次优的表现。此外，在模型训练中，有无硬标签对性能的影响很小。此后的实验都将保持硬标签的存在。
- **Ablation on training hyper-parameters**. 
为了确保全面公正的比较，我们使用策略“A2”对DKD、DIST和原始KD进行了不同学习率和权重衰减的各种配置的调查。结果显示在**表5**中，揭示了一个有趣的发现：原始KD在所有配置下表现优于其他方法。另外，值得注意的是，原始KD在不同学习率和权重衰减配置下始终表现接近最佳，展示了其鲁棒性和在各种设置下保持竞争性结果的能力。

### 3.5 Distillation results with different teacher-student pairs
每种方法选择性能最高的配置，以确保公平比较。相应的结果如**表6**所示。与之前的实验结果一致，与DKD和DIST相比，原始KD在所有组合中都能达到同等或略好的性能。这些结果证明了原始KD在各种师生组合中实现最先进表现的潜力。
![](图片/知识蒸馏/VanillaKD%20Revisit/tab6.png)
- **Marginal gains when teacher is sufficiently large.**
虽然在我们的实验中，原始KD在各种架构下可以实现最先进的结果，但它并不没有局限性。例如，我们观察到使用BEiTv2-B教师训练的Res50学生模型的结果与使用更大的BEiTv2-L训练的结果相似甚至更好。我们评估了BEiTv2-B教师和BEiTv2-L教师对于所有基于logits的基准模型的影响，并使用了三种不同的阶段配置。如**表9**所示，结果显示随着教师模型容量的增加，性能逐渐下降。这表明原始KD无法从更大的教师模型中获得益处，即使在大规模数据集上进行训练。
![](图片/知识蒸馏/VanillaKD%20Revisit/tab9.png)
- **Sustained benefits when enlarging the teacher’s training set**.
我们还评估了用于教师模型的训练集对学生性能的影响。我们比较了两个BEiTv2-B教师模型：一个在ImageNet-1K上预训练，另一个在ImageNet-21K上预训练。随后，我们使用策略“A1”训练两个不同的学生模型，进行了1200个时期的训练。评估结果如表7所示。结果清楚地表明，当教师模型在更大规模的数据集上进行训练时，它对学生的性能产生积极影响。我们假设在更大的数据集上预训练的教师模型对数据分布有更好的理解，从而促进学生的学习并提高性能。![](图片/知识蒸馏/VanillaKD%20Revisit/tab7.png)

### 3.6 Exploring the power of vanilla KD with longer training schedule
训练足够长时间能让模型精度进一步提高，甚至达到SOTA性能。（ResNet-50, ViT-S, and ConvNeXtV2-T）
![](图片/知识蒸馏/VanillaKD%20Revisit/tab8.png)
### 3.7 Comparison with masked image modeling
结果如表10所示，突出了原始KD在训练出色模型方面的效率。具体而言，即使在仅300个时期的训练时间内，原始KD也实现了84.42%的准确率，超过MIM模型0.53%，同时仅消耗了训练时间的五分之一。
此外，当将训练时期延长到1200个时期时，原始KD实现了85.03%的性能，超过最佳表现的MIM训练模型1.14%。这些结果进一步证明了与MIM框架相比，原始KD的有效性和时间效率。
![](图片/知识蒸馏/VanillaKD%20Revisit/tab9.png)

### 3.8 Transferring to downstream task
为了评估学生在ImageNet上提取的迁移学习性能，我们使用COCO基准进行了对象检测和实例分割任务的实验。我们采用Res50和ConvNeXtV2-T作为主干，并用提取的检查点对它们进行初始化。我们实验中常用的检测器是Mask RCNN和Cascade Mask RCNN。
表11报告了相应的结果。当使用我们蒸馏后的Res50模型作为骨干网络时，Mask RCNN在“1×”时间表下的框AP上超过了使用从头训练的骨干网络的最佳对应模型3.1%，在“2×”时间表下超过了2.1%。同样，在使用ConvNeXtV2-T作为骨干网络的情况下，使用原始KD训练的模型始终可以提供更好的性能。这些结果证明了原始KD所实现的性能改进在下游任务中的高效可转移性。![](图片/知识蒸馏/VanillaKD%20Revisit/tab11.png)