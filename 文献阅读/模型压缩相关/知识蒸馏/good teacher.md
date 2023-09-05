 **Knowledge distillation  A good teacher is patient and consistent.** *L. Beyer, Xiaohua Zhai, Amélie Royer, L. Markeeva, Rohan Anil, Alexander Kolesnikov.* **Computer Vision and Pattern Recognition, 2021** [(PDF)](../../Notetool/papers/CVPR%202022Knowledge%20distillation%20%20A%20good%20teacher%20is%20patient%20and%20consistent.pdf 2022Knowledge distillation  A good teacher is patient and consistent.pdf>)  [(arxiv)](https://arxiv.org/abs/2106.05237)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F97d8823ca3c9bd932cec8ad6f3b194168e7cec92%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/97d8823ca3c9bd932cec8ad6f3b194168e7cec92)
#### 引言
首先，对于非常大的教师模型，我们可能会尝试在离线环境中预先计算图像的教师激活值，以节省计算资源。然而，正如我们将要展示的，这种固定教师模型的方法效果并不好。
其次，知识蒸馏不仅仅局限于模型压缩的领域，在其他不同的应用场景下也被广泛使用。不同的研究论文可能会对知识蒸馏的设计选择提出不同甚至相反的建议。在图2中可以看到这种情况。
第三，知识蒸馏通常需要更多的训练周期才能达到最佳性能，远远超过常规的监督训练所需的周期数。这是因为知识蒸馏涉及将教师模型的知识传递给学生模型，这可能需要更多的迭代才能达到最佳性能。
最后，需要注意的是，在常规长度的训练中可能看起来次优的选择往往在长时间运行中表现最佳，反之亦然。因此，在选择知识蒸馏的策略和方法时，需要进行实验和评估，以找到适合特定任务和训练周期的最佳方法。
#### 实验设置
- 数据集：flowers102 , pets , food101, sun397  and ILSVRC-2012 (“ImageNet”) 
- 评价指标：图片分类精度
- 教师网络： BiT-M-R152x2  学生网络：BiT-ResNet-50
- 蒸馏损失：$K L ( p _ { t } | | p _ { s } ) = \sum _ { i\in c } [ - p _ { t , i } l o g p _ { s , i } + p _ { t , i } | \log p _ { t , i } |$
- 使用余弦学习率衰减，不需要热身重启（warm restarts ）；使用 “decoupled” weight decay方法；使用梯度截断（ gradient clipping）去稳定训练，将梯度的全局L2范数阈值设定为1.0。
- batch size :512(除了在imagenet为4096)
- 使用了“agressive” mixing coefficients(mixup ) 数据增强
- 使用了“inception-style” 裁剪、并且讲图片
### 模型压缩蒸馏
##### 一致的重要性
四类可选框架：
![](图片/知识蒸馏/cvpr%202022%20good%20teacher%20is%20patient%20and%20consistent/good1.png)
- fixed teacher ：
	1. fix/rs: 将图片直接重整为$224^2$px 输入给教师和学生网络
	2. fix/cc：教师网络使用fixed central 裁剪，学生使用 温和随机（mild random ）裁剪
	3. fix/ic_ens 教师网络使用一种heavy数据增强方法，该方法使用1k inception 裁剪；学生网络使用随机 inception 裁剪。
	后两种类似于给学生网络输入信号加噪声的“noisy student”
- 独立噪声：
	1. ind/rc : 分别给教师网络和学生网络计算mild random裁剪
	2. ind/ic : 使用 更大幅度（heavier）inception 裁剪
-  一致指导(Importance of “consistent” teaching)：
	1. same/rc：轻微的随机裁剪
	2. same/ic ：heavy Inception裁剪
	（仅对图像进行一次随机裁剪，并同时输入进学生网络和教师网络）
- 函数匹配（Function matching.） ：
	拓展了一致指导，通过混合两个输入图像来 扩展图像的输入流形同时输入进学生网络和教师网络。简称“FunMatch”。
下图展示了在Flowers102数据集上进行了10,000个epoch的训练曲线，涵盖了上述所有configurations。这些结果清楚地表明，“一致性（consistency）”是关键：所有“不一致(inconsistent)”的蒸馏设置都在较低的精度处停滞，而一致的设置显著提高了学生模型的性能，其中函数匹配方法效果最好。此外，训练损失显示，对于如此小的数据集，使用固定的教师模型会导致严重的过拟合。相反，函数匹配（FunMatch）方法在训练集上从未达到如此高的loss，同时更好地推广到验证集上。
![](图片/知识蒸馏/cvpr%202022%20good%20teacher%20is%20patient%20and%20consistent/good2.png)

##### 耐心（patient）的重要性
一种观点认为蒸馏是监督学习的变体，因为蒸馏标签是由教师网络提供的。但这样继承了监督学习的所有缺点，例如激进的数据增强可能会扭曲真是标签，而较少的数据增强会产生过拟合。
然而，如果我们从函数匹配的角度解释蒸馏，并且关键地保证教师和学生网络输入的一致性。这样就可以使用激进的数据增强，甚至在图像已经非常扭曲时，依旧可以促进相关函数的匹配。无需考虑由于限制数据增强而导致的过拟合问题。
在下图中，对于每个数据集，展示了根据验证集选出的最佳函数匹配学生模型在不同训练周期内的测试准确率的变化。教师模型以红线表示，学生模型并最终达到了其准确率，但所需的训练周期比在监督训练设置中使用的要多得多。关键是，即使我们进行了100万个训练周期的优化，也没有出现过拟合的情况。
![](图片/知识蒸馏/cvpr%202022%20good%20teacher%20is%20patient%20and%20consistent/good3.png)
- 扩大至imagenet
如下图：fixed teacher在600次之后开始过拟合。相比之下，一致性教学方法随着训练持续时间的增加不断提高性能。基于此，我们可以得出结论，一致性是使蒸馏方法在ImageNet上奏效的关键，类似于前面讨论过的小型和中型数据集上的行为。
与简单的一致性教学相比，函数匹配在短期训练计划下的表现稍差，这可能是由于欠拟合造成的。但是，当我们增加训练计划的长度时，函数匹配的改进变得明显：例如，仅使用1200个训练周期，它就能够达到一致性教学4800个训练周期的性能，因此节省了75%的计算资源。
![](图片/知识蒸馏/cvpr%202022%20good%20teacher%20is%20patient%20and%20consistent/good4.png)
- 在不同的分辨率下蒸馏
	具有高分辨率输入的教师能训练出更小、更快的学生
- 二阶预处理器能提高训练效率
	shampoo优化器能加速训练
- 一个好的初始化能极大的缩减训练时长，并且在较短训练时间下有更高的精度。但是当训练时间足够长时这一优势缩小。
- 不同网络间蒸馏
	使用224px+384px两种分辨率下的BiT-M-R152x2教师网络对mobileNet学生网络进行训练，在imagenet上达到了sota精度。
- 在域外（out-of-domain）数据上蒸馏
	通过将知识提取视为“函数匹配”，可以得出一个合理的假设，即提取可以在任意图像输入上进行。。首先，我们观察到使用域内数据进行提取效果最好。令人惊讶的是，即使图像完全无关，蒸馏依然在一定程度上有效，尽管精度有所下降。
![](图片/知识蒸馏/cvpr%202022%20good%20teacher%20is%20patient%20and%20consistent/good5.png)
为了确保我们观察到的sota蒸馏结果不是我们精心调整的训练设置的产物，即非常长的训练计划和激进的mixup增强，我们训练相应的基准ResNet-50模型。
具体而言，我们在没有蒸馏损失的情况下，重新利用了蒸馏训练设置在ImageNet数据集上进行监督训练。为了进一步加强我们的基准模型，我们额外尝试了具有动量的SGD优化器，这在ImageNet上通常比Adam优化器效果更好。结果如上图7所示。观察到仅使用标签进行训练且没有蒸馏损失会导致显着更差的结果，并在长时间训练计划中开始出现过拟合。因此，蒸馏是使训练取得良好效果的必要条件。
#### 结论
与提出一种新的模型压缩方法不同，我们详细研究了现有的常见的知识蒸馏过程，并确定了如何在模型压缩的背景下使其发挥出色。我们的关键发现源于对知识蒸馏的特定解释：我们建议将其视为函数匹配任务。这不是知识蒸馏的典型观点，通常它被视为“一个强大的教师生成更好的（软）标签，这对于训练一个更好、更小的学生模型是有用的”，即监督学习的变体。
我们同时采用了三个要素：（i）确保教师和学生始终获得相同的输入，包括噪声，（ii）引入激进的数据增强来丰富输入图像流形（通过mixup），以及（iii）使用非常长的训练计划。
尽管我们的方法中每个部分看起来很简单，但我们的实验证明，必须同时应用它们才能获得最佳结果。我们在将非常大的模型压缩到更实用的ResNet-50架构方面取得了非常强大的实证结果。我们相信，这些结果从实际角度非常有用，并为将来关于压缩大规模模型的研究提供了非常强大的基线模型。