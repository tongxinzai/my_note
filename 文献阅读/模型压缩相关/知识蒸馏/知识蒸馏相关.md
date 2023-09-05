### KD
**Distilling the Knowledge in a Neural Network.** *Geoffrey E. Hinton, Oriol Vinyals, J. Dean.* **arXiv.org, 2015** [(PDF)](../../Notetool/papers/Distilling%20the%20Knowledge%20in%20a%20Neural%20Network.pdf the Knowledge in a Neural Network.pdf>)  [(arxiv)]()[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0c908739fbff75f03469d13d4a1a07de3414ee19%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/0c908739fbff75f03469d13d4a1a07de3414ee19)

最原始的KD，提出使用softmax“软化 ”概率分布。
$$q_i=\cfrac{exp(z_i/T)}{\sum_jexp(z_j/T)} \tag{1}$$
$$\begin{array}{c}
L=\lambda L_{hard}+(1-\lambda)T^2 L_{soft}\\
L_{soft}=-\sum_j^N p^T_j\log(q^T_j) \\
L_{hard}=-\sum_j^N c_j\log(q^1_j)
\end{array}$$
需要注意的是，在同时使用soft target和hard target的时候，需要在soft target之前乘上$T^{2}$的系数，这样才能保证soft target和hard target贡献的梯度量基本一致。 


### DeiT
- **Training data-efficient image transformers & distillation through attention.** *Hugo Touvron, M. Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Herv'e J'egou.* **International Conference on Machine Learning, 2020** [(PDF)](../../Notetool/papers/Training%20data-efficient%20image%20transformers%20&%20distillation%20through%20attention.pdf data-efficient image transformers & distillation through attention.pdf>)  [(arxiv)](https://arxiv.org/abs/2012.12877)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fad7ddcc14984caae308c397f1a589aae75d4ab71%3Ffields%3DcitationCount)]  [(code)](https://github.com/facebookresearch/deit.)
>DeiT 是一个全 Transformer 的架构。其核心是提出了针对 ViT 的教师-学生蒸馏训练策略，并提出了 token-based distillation 方法，使得 Transformer 在视觉领域训练得又快又好。
![](图片/知识蒸馏/deit/deit1.png)
就是在vit的基础上加了一个蒸馏损失，定义了软蒸馏和硬蒸馏：
$$\mathcal{L}_{\text {global }}=(1-\lambda) \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{\mathrm{s}}\right), y\right)+\lambda \tau^{2} \mathrm{KL}\left(\psi\left(Z_{\mathrm{s}} / \tau\right), \psi\left(Z_{\mathrm{t}} / \tau\right)\right)$$
$$\mathcal{L}_{\text {global }}^{\text {hardDistill }}=\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y\right)+\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y_{\mathrm{t}}\right)$$
### DIST
- **Knowledge Distillation from A Stronger Teacher.** *Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu.* **Neural Information Processing Systems, 2022** [(PDF)](<../../NoteTool/papers/Knowledge Distillation from A Stronger Teacher.pdf>)  [(arxiv)](http://arxiv.org/abs/2205.10536)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc0ae5848ba0141dd3f827321f46110f52946764b%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/c0ae5848ba0141dd3f827321f46110f52946764b) [(code)](https://github.com/hunto/DIST_KD)

**DIST**这篇文章认为学生与教师差距过大的时候，KL散度的精确匹配会影响训练效果，所以现有的方法效果不佳。因此作者提出了一种简单保留教师与学生之间关系的方法，并根据相关性明确教师网络中的类间关系。此外，考虑到不同实例与每个类别之间具有不同的语义相似度，还将这种关系匹配扩展到类内级别。
主要是用皮尔逊相关系数$ρ_p​(u;v)$去代替KL散度来衡量类内和类间实例的关系：
$$d _ { p } ( u , v ) : = 1 - p _ { p } ( u , v )$$
$$L_{inter} := \frac {1}{B} \sum_{i=1}^B d_p(Y^{(s)}_{i,:} , Y^{(s)}_{i,:})$$
$$L_{intra} := \frac {1}{C} \sum_{i=1}^C d_p(Y^{(s)}_{:,j} , Y^{(s)}_{:,j})$$
$$L_{tr} = αL_{cls} + βL_{iner} + γL_{intra}$$
### DKD
- **Decoupled Knowledge Distillation.** *Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang.* **Computer Vision and Pattern Recognition, 2022** [(PDF)](../../Notetool/papers/Decoupled%20Knowledge%20Distillation.pdf Knowledge Distillation.pdf>)  [(arxiv)](https://arxiv.org/abs/2203.08679)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F72b989a52a5cc2eee44bba29e8d225ce7bc07666%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/72b989a52a5cc2eee44bba29e8d225ce7bc07666) [(code)](https://github.com/megvii-research/mdistiller)

**DKD**提供了一个深入的视角来研究logit蒸馏，将经典的知识蒸馏（KD）方法划分为目标类别知识蒸馏（TCKD）和非目标类别知识蒸馏（NCKD）两个部分，并分别分析和证明了这两个部分的影响。
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


###  good teacher
 **Knowledge distillation  A good teacher is patient and consistent.** *L. Beyer, Xiaohua Zhai, Amélie Royer, L. Markeeva, Rohan Anil, Alexander Kolesnikov.* **Computer Vision and Pattern Recognition, 2021** [(PDF)](../../Notetool/papers/CVPR%202022Knowledge%20distillation%20%20A%20good%20teacher%20is%20patient%20and%20consistent.pdf 2022Knowledge distillation  A good teacher is patient and consistent.pdf>)  [(arxiv)](https://arxiv.org/abs/2106.05237)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F97d8823ca3c9bd932cec8ad6f3b194168e7cec92%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/97d8823ca3c9bd932cec8ad6f3b194168e7cec92) [(code)](https://github.com/google-research/big_vision)

这篇文章主要是想说训练一个学生模型需要更长的epoch，并且学生网络与教师网络的输入要保持一致。这就是“patient”和"consistent"。

一个重要的观点是将知识蒸馏的过程看作是学生模型对教师模型进行”函数匹配“的过程，而不是一种由教师模型提供标签的监督学习过程。这样就可以使用较好的数据增强而不用担心由于长时间训练而导致的过拟合问题，能更好的训练学生网络。

### VanillaKD Revisit
**VanillaKD  Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale.** *Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang.* **arXiv.org, 2023** [(PDF)](<../../NoteTool/papers/VanillaKD  Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale.pdf>)  [(arxiv)](https://arxiv.org/abs/2305.15781)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fda1946bb4220e743e8f46946397a9b31e609df74%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/da1946bb4220e743e8f46946397a9b31e609df74) [(code)](https://github.com/Hao840/vanillaKD)

![](../图片/知识蒸馏/VanillaKD%20Revisit/fig1.png)
原始KD的能力被轻视了，作者在大数据集上训练原始KD以证明原始KD方法简介有效。因为现在很多蒸馏模型都基于较小的数据集例如cifar-100上做评估，而随着数据集的增加，原始蒸馏方法与现有的先进方法的差距会逐渐减小i。
在合适的数据增强与较长的训练时间的加持下，原始KD在精度和稳健性方面不输现有的先进方法，甚至在个别方面能达到SOTA。

### SimKD
- **Knowledge Distillation with the Reused Teacher Classifier.** *Defang Chen, Jianhan Mei, Hailin Zhang, C. Wang, Yan Feng, Chun Chen.* **Computer Vision and Pattern Recognition, 2022** [(PDF)](<../../NoteTool/papers/Knowledge Distillation with the Reused Teacher Classifier.pdf>)  [(arxiv)](https://arxiv.org/abs/2203.14001)[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F5eafb52964f99514ae04952e3dceb63a22b3ec2f%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/5eafb52964f99514ae04952e3dceb63a22b3ec2f)  [(code)](https://github.com/DefangChen/SimKD)
![[../图片/知识蒸馏/SimKD/fig1.png]]
作者主要是用一个投影来提取学生网络的特征，接着用L2损失函数相与教师网络匹配：  $$L _ { S i m K D }  = | | \mathbf f ^ { t } - P ( \mathbf f ^ { s } ) | | _ { 2 } ^ { 2 }\tag{3}$$其中，投影仪P(·)被设计为以相对较小的成本匹配特征维度，同时又能有效地保证准确的对齐。但是当特征维度不匹配时，这需要一个投影仪，从而增加了模型的复杂性。作者认为大多数能力特定的信息都包含在深层中，并期望重新使用这些层，重用这些层的作用被反复强调。通常情况下，重用更多的层会导致更高的学生准确率，但会带来推理的负担增加。  