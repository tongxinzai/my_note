- **Decoupled Knowledge Distillation.** *Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang.* **Computer Vision and Pattern Recognition, 2022** [(PDF)](../../Notetool/papers/Decoupled%20Knowledge%20Distillation.pdf Knowledge Distillation.pdf>)  [(arxiv)](https://arxiv.org/abs/2203.08679)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F72b989a52a5cc2eee44bba29e8d225ce7bc07666%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/72b989a52a5cc2eee44bba29e8d225ce7bc07666)
## Abstrct
当前最先进的蒸馏方法主要基于从中间层中提取深层特征进行蒸馏，而对于logit蒸馏的重要性往往被大大忽视。为了提供一种新颖的视角来研究logit蒸馏，我们将经典的知识蒸馏损失重新定义为两个部分，即目标类别知识蒸馏（TCKD）和非目标类别知识蒸馏（NCKD）。我们通过实证研究和证明了这两个部分的影响：TCKD传递了关于训练样本“难度”的知识，而NCKD则是logit蒸馏有效的主要原因。更重要的是，我们揭示了经典的知识蒸馏损失是一种耦合形式，它抑制了NCKD的效果，并限制了平衡这两个部分的灵活性。为了解决这些问题，我们提出了解耦合知识蒸馏（DKD），使TCKD和NCKD能够更高效、更灵活地发挥作用。
与复杂的基于特征的方法相比，我们的DKD在CIFAR-100、ImageNet和MSCOCO数据集上实现了可比甚至更好的结果，并具有更好的训练效率，适用于图像分类和目标检测任务。本文证明了logit蒸馏的巨大潜力，希望对未来的研究有所帮助。
## introduction
最初的KD发展与最小化teacher和student模型预测logit的KL散度，如图1所示。但是随着基于深度特征蒸馏的发展，logit蒸馏则慢慢淡出视野，因为特征蒸馏的效果相对更好。但是同时也带来一些问题，其带来的额外计算量（模型结构和复杂的操作）不能满足实际需求。
logit蒸馏则需求很小的计算量和存储，但是其表现不及特征蒸馏。但是从直觉来说logit方法应该更加有效，因为其使用的是更加高维的语义特征，为了发现问题，作者将经典的KD损失分解为两部分：（1）将目标类和其他非目标类定义为二分类预测；（2）所有的非目标类构成多分类预测。
总的来说，我们的贡献可以总结如下：
•我们提供了一个深入的视角来研究logit蒸馏，将经典的知识蒸馏（KD）方法划分为目标类别知识蒸馏（TCKD）和非目标类别知识蒸馏（NCKD）两个部分，并分别分析和证明了这两个部分的影响。
•我们揭示了经典KD损失的局限性，这是由于其高度耦合的形式所导致的。将NCKD与教师模型的置信度耦合在一起会压制知识传递的效果。将TCKD与NCKD耦合在一起则限制了平衡这两个部分的灵活性。
•我们提出了一种名为DKD的有效logit蒸馏方法，以克服这些限制。DKD在各种任务上实现了最先进的性能。我们还通过实验证实了与基于特征的蒸馏方法相比，DKD具有更高的训练效率和更好的特征可迁移性。
以上是我们的主要贡献。
## Rethinking Knowledge Distillation
给定分类概率为$\textbf{P}=[p_{1},p_{2},...,p_t,...,p_C]\in R^{1 \times C}$,则softmax计算为：
$$p_i=\frac{exp(z_i)}{\sum_{j=1}^{C}exp(z_j)}\tag{1}
$$
为了将上述公式分解为目标相关和目标无关的部分，给出如下定义，$\bf{b}=[p_t,p_{\backslash \ t}]\in R^{1 \times 2}$,计算如下：
同时，声明$\bf{P}=[\hat{p}_{1},...,\hat{p}_{t-1},\hat{p}_{t+1}...,\hat{p}_C]\in R^{1\times(C-1)}$独立进行非目标类的建模，计算如下：
$$\hat p_{i}=\frac{exp(z_{i})}{\sum_{j=1,j\ne i}^{c}exp(z_j)}\tag{2}
$$
定义T和S为teacher和student模型，则经典的使用KL散度计算的KD损失为：
$$\left.\begin{array}{l}{KD=KL(p^{T}||p^{S})}\\
{=p_{t}^{T}\log(\frac{p_{t}^{T}}{p_{t}^{S}})+\sum_{i=1,i\ne t}^{c}p_{i}^{T}\log(\frac{p_{i}^{T}}{p_{i}^{s}})}\end{array}\right.\tag{3}$$
根据之前两式有$\hat p_{i}^{T}=p_{i}/p_{\backslash \ t}$可以将上式改写为：
$$\left.\begin{array} {c}
KD=p_{t}^{T}\log(\frac{p_{t}^{T}}{p_{t}^{S}})+
p_{\backslash\ t}^{T}\sum_{i=1,i\ne t}^{c}\hat p_{i}^{T}(\log(\frac{\hat p_{i}^{T}}{\hat p_{i}^{S}})+ 
\log(\frac{p_{\backslash\ t}^{T}}{p_{\backslash\ t}^{S}}))
\\
=\underbrace{p_{t}^{T}\log(\frac{p_{t}^{T}}{p_{t}^{S}})+ 
p_{\backslash\ t}^{T}\log(\frac{p_{\backslash\ t}^{T}}{p_{\backslash\ t}^{S}})}_{KL(b^{T}||b^{S})}+
\underbrace{ p_{\backslash\ t}^{T}\sum_{i=1,i\ne t}^{c} \hat p_{i}^{T}\log(\frac{p_{i}^{T}}{p_{i}^{S}})}_{KL(\hat p^{T}||\hat p^{S})}
\end{array}\right.\tag{4}$$
于是公式（4）可以简化为：
$$\left.\begin{array}{l}{KD=KL(b^{T}||b^{S})+(1 - p_{ t}^{T})KL(\hat p^{T}||\hat p^{S})}  \end{array}\right.\tag{5}$$
再改写成TCKD和NCKD的形式为：
$$\left.\begin{array}{l}{KD=TCKD+(1 - p_{\backslash\ t}^{T})NCKD}  \end{array}\right.\tag{6}  $$
### 3.2 Effects of TCKD and NCKD
  
实验设计：数据集为CIFAR-100. ResNet， WideResNet（WRN）和ShuffleNet被用作训练模型，相同结构和不同结构的网络都被考虑到。实验结果如表1所示。
![](图片/知识蒸馏/DKD/tab1.png)
单用TCKD不一定有效，但是单用NCKD反而效果可能比经典的KD要好。
**TCKD迁移了关于训练样本“难度”的知识**.由于TCKD传达了训练样本的“难度”，我们认为当训练数据变得具有挑战性时，有效性就会显现出来。因此训练数据越困难，TCKD可以提供的好处就越多
（1） 应用强增强是增加训练数据难度的一种简单方法。
（2） 噪声标签也会增加训练数据的难度。
（3） 还考虑了具有挑战性的数据集（例ImageNet）
![](图片/知识蒸馏/DKD/tab2.png)
**NCKD是logit蒸馏工作的突出原因，但被大大抑制。**
有趣的是，我们在表1中注意到，当仅应用NCKD时，其性能与经典KD相当，甚至更好。结果表明，非目标类之间的知识对logit的提炼至关重要，它可以成为突出的“暗知识”。然而根据方程（5），我们注意到NCKD损失与$(1−p_{t}^{T})$有关，其中$p_{t}^{T}$表示教师对目标分类的confidence 。因此，更有confidence的预测会导致更小的NCKD权重。我们认为，教师对培训样本越有信心，就越能提供可靠和有价值的知识!
## Decoupled Knowledge Distillation（DKD）
到目前为止，我们已经将经典的KD损失重新表述为两个独立部分的加权和，并进一步验证了TCKD的有效性，揭示了NCKD的抑制。具体来说，TCKD转移了关于训练样本难度的知识。对于更具挑战性的训练数据，TCKD可以获得更显著的改进。NCKD在非目标类之间传递知识，当权重$(1−p_{t}^{T})$相对较小时，NCKD会被抑制。
![](图片/知识蒸馏/DKD/tab5.png)
从直觉上来说，TCKD和NCKD都是不可或缺的，至关重要的。而在经典KD公式中，TCKD和NCKD从以下几个方面进行耦合：
- 一方面，NCKD与$(1−p_{t}^{T})$相结合，可以在预测良好的样本上抑制NCKD。由于表5的结果表明，预测良好的样本可以带来更多的性能增益，耦合形式可能会限制NCKD的有效性。
- 另一方面，在经典KD框架下，NCKD和TCKD的权值是耦合的。不能为了平衡重要性而改变每一项的比重。我们认为TCKD和NCKD应该分开考虑，因为它们的贡献来自不同的方面。
因此将经典的KD损失分解为TCKD，NCKD之后我们引入两个超参数 α\\alpha\\alpha 和β\\beta\\beta ，形成DKD的损失函数：
$$\left.\begin{array}{l}{KD=\alpha TCKD+\beta NCKD}  \end{array}\right. \tag{7} $$
在上式中，可以调整 αa和 β 来平衡两项，通过解耦TCKD和NCKD，DKD可以实现更好的logit蒸馏。
## Main Results
![](图片/知识蒸馏/DKD/tab6.png)
![](图片/知识蒸馏/DKD/tab7.png)
![](图片/知识蒸馏/DKD/tab10.png)
如上表格所示，本文在CIFAR-100、ImageNet和MC-COCO数据集上进行了实验，实验结果表明DKD方法具有先进的水平。
![](图片/知识蒸馏/DKD/tab11.png)
表11和表12中的实验结果一致表明，我们的DKD缓解了更大的模型并不总是更好的教师问题
![](图片/知识蒸馏/DKD/fig2.png)
我们评估了最先进的蒸馏方法的训练成本，证明了DKD的高培训效率。如图2所示，我们的DKD在模型性能和训练成本（例如，训练时间和额外参数）之间实现了最佳权衡。由于DKD是从经典KD重新公式化的，它需要与KD几乎相同的计算复杂度，当然也不需要额外的参数。然而，基于特征的提取方法需要额外的训练时间来提取中间层特征，以及GPU内存成本。
![](图片/知识蒸馏/DKD/fig3.png)
（1） t-SNE（图3）结果表明，DKD的表示比KD更可分离，证明DKD有利于深层特征的可分辨性。（2） 我们还可视化了学生和教师logits的相关矩阵的差异（图4）。
## 结论
本文通过将经典的KD损失分解为目标类知识精馏(TCKD)和非目标类知识精馏(NCKD)两部分，为解释logit精馏提供了一种新的观点。分别对两部分的效果进行了研究和验证。更重要的是，揭示了KD的耦合公式限制了知识转移的有效性和灵活性。为了克服这些问题，提出了解耦知识蒸馏(DKD)，它在CIFAR-100、ImageNet和MS-COCO数据集上实现了图像分类和目标检测任务的显著改进。此外，还证明了DKD算法在训练效率和特征可转移性方面的优越性。希望本文对未来的logit精馏研究有所贡献。
局限性和未来的作品。下面讨论了值得注意的局限性。在目标检测任务上，DKD不能超过最先进的基于特征的方法(例如ReviewKD\[1\])，因为基于logit的方法不能传递关于定位的知识。此外，还为如何调整β提供了直观的指导。然而，β与精馏性能之间的严格相关性尚未得到充分研究，这将是未来的研究方向。