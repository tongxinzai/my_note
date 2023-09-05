**Distilling the Knowledge in a Neural Network.** *Geoffrey E. Hinton, Oriol Vinyals, J. Dean.* **arXiv.org, 2015** [(PDF)](../../Notetool/papers/Distilling%20the%20Knowledge%20in%20a%20Neural%20Network.pdf the Knowledge in a Neural Network.pdf>)  [(arxiv)]()[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0c908739fbff75f03469d13d4a1a07de3414ee19%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/0c908739fbff75f03469d13d4a1a07de3414ee19)
## distillation

神经网络通常会使用softmax产生一个概率分布，$z_i$代表计算出每类的概率，$q_i$代表$z_i$和其他分数的比较。$$q_i=\cfrac{exp(z_i/T)}{\sum_jexp(z_j/T)} \tag{1}$$T通常设置为1。采用更大的T，会产生更加软化（softer）的概率分布。首先通过这种方法生成软标签，然后将软标签和硬标签同时用于新网络的学习。

###  Matching logits is a special case of distillation

相对于蒸馏模型的每个logit$z_i$，传递集中的每种情况都贡献了一个交叉熵梯度$dC / dz_i$。如果笨重模型具有产生软目标概率$p_i$的logit $v_i$，并且在温度为 T 的条件下进行传递训练，则该梯度由下式给出：
$$\frac { \partial C } { \partial z _ { i } } = \frac { 1 } { T } ( q _ { i } - p _ { i } ) = \frac { 1 } { T } \left( \frac { e ^ { z _ { i } / T } } { \sum _ { j } e ^ { z _ { j } / T } } -\frac { e ^ { v _ { i } / T } } { \sum _ { j } e ^ { v _ { j } / T } } \right)$$
如果温度比对数的幅度高，我们可以近似得出：
$$\frac { \partial C } { \partial z _ { i } } \approx \frac { 1 } { T } ( \frac { 1 + z _ { i } / T } { N + \sum _ { j } z_j / T } - \frac { 1 + v _ { i } / T } { N + \sum _ { j } v _ { j } / T } )$$
我们假定logits的均值为0，则$\sum_jz_j=\sum_jv_j=0$，上式可以简化为：
$$\frac { \partial C } { \partial z _ { i } } \approx \frac { 1 } { N T ^ { 2 } } ( z _ { i } - v _ { i } )$$
因此，在高温下，如果logits是零均值的，则蒸馏等效于最小化均方误差：$1/2(z_i -v_i)^2$。当T较小时，蒸馏更加关注负标签，在训练复杂网络的时候，这些负标签是几乎没有约束的，这使得产生的负标签概率是噪声比较大的，所以采用更大的T值（上面的简化方法）是有优势的。而另一方面，这些负标签概率也是包含一定的有用信息的，能够用于帮助简单网络的学习。这两种效应哪种占据主导，是一个实践问题
###  soft targets

训练大数据集的一个简单方法是集成模型（将数据分成数个子集，分别训练然后集成），这种方法很容易并行化，但是却在测试的时候需要耗费大量的计算资源，而distillation可以解决这个问题。
集成的一个主要问题是容易过拟合，这里利用soft targets来处理这个问题：
$$K L ( p ^ { g } , q ) + \sum _ { m \in A _ { k } } K L ( p ^ { m } , q ) $$
##  蒸馏过程
蒸馏过程的目标函数由distill loss(对应soft target)和student loss(对应hard target)加权得到。$$\begin{array}{c}
L=\lambda L_{hard}+(1-\lambda)T^2 L_{soft}\\
L_{soft}=-\sum_j^N p^T_j\log(q^T_j) \\
L_{hard}=-\sum_j^N c_j\log(q^1_j)
\end{array}$$，其中$$p^T_i=\frac{\exp(v_i/T)}{\sum_k^N \exp(v_k/T)} , q^T_j=\frac{\exp(z_j/T)}{\sum_k^N \exp(z_k/T)},q^1_j=\frac{\exp(z_j)}{\sum_k^N \exp(z_k)}$$- S在T=1的条件下的softmax输出和ground truth的cross entropy就是**Loss函数的第一部分**$L_{hard}$ 。
- $v_i$: Net-T的logits
- $z_i$: Net-S的logits
- $p^T_i$: Net-T的在温度=T下的softmax输出在第i类上的值
- $q^T_i:$ Net-S的在温度=T下的softmax输出在第i类上的值
- $c_i$: 在第i类上的ground truth值, $c_i\in\{0,1\},$ 正标签取1，负标签取0.
- N: 总标签数量
- Net-T 和 Net-S同时输入 transfer set (这里可以直接复用训练Net-T用到的training set), 用Net-T产生的softmax distribution (with high temperature) 来作为soft target，Net-S在相同温度T条件下的softmax输出和soft target的cross entropy就是**Loss函数的第一部分**$L_{soft}$
- 第二部分$L_{hard}$ 的必要性其实很好理解: T也有一定的错误率，使用ground truth可以有效降低错误被传播给S的可能。打个比方，老师虽然学识远远超过学生，但是他仍然有出错的可能，而这时候如果学生在老师的教授之外，可以同时参考到标准答案，就可以有效地降低被老师偶尔的错误“带偏”的可能性。

**【讨论】**
- 实验发现第二部分所占比重比较小的时候，能产生最好的结果，这是一个经验的结论。一个可能的原因是，由于soft target产生的gradient与hard target产生的gradient之间有与 T 相关的比值。原论文中只是一笔带过，下面补充了一些简单的推导。
- **Soft Target:**$L_{soft}$

$$L_{soft}=-\sum_j^N p^T_j\log(q^T_j)=-\sum_j^N \frac{z_j/T\times\exp(v_j/T)}{\sum_k^N \exp(v_k/T)}\left(\frac{1}{\sum_k^N \exp(z_k/T)}-\frac{\exp (z_j / T) }{\left( \sum_k^N \exp(z_k/ T)\right) ^ 2}\right)$$

$$\approx -\frac{1}{T\sum_k^N \exp(v_k/T)}\left(\frac{\sum_j^Nz_j\exp(v_j/T)}{\sum_k^N \exp(z_k/T)}-\frac{\sum_j^N z_j\exp (z_j/ T)\exp(v_j/T) }{\left( \sum_k^N \exp(z_k / T)\right) ^ 2} \right)$$

- **Hard Target:$L_{hard}$

$$L_{hard}=-\sum_j^N c_j\log(q^1_j)=-\left(\frac{\sum_j^N c_jz_j }{ \sum_k^N \exp(z_k )}-\frac{\sum_j^N c_jz_j\exp (z_j) }{\left( \sum_k^N \exp(z_k)\right) ^ 2} \right)$$

- 由于 $\frac{\partial L_{soft}}{\partial z_i}$的magnitude大约是 $\frac{\partial L_{hard}}{\partial z_i}$ 的 $\frac{1}{T^2}$ ，因此在同时使用soft target和hard target的时候，需要在soft target之前乘上$T^{2}$的系数，这样才能保证soft target和hard target贡献的梯度量基本一致。 