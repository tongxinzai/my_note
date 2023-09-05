在PyTorch中，张量（Tensor）是一种多维数组，类似于Numpy数组。PyTorch提供了丰富的张量基本运算功能，下面是一些常见的张量基本运算示例：
1. 张量创建：
	   1. torch.tensor(data)：根据给定的数据创建张量，可以自动推断数据类型。
	   2. torch.zeros(shape)：创建指定形状的全零张量。
	   3. torch.ones(shape)：创建指定形状的全一张量。
	   4. torch.eye(n)：创建一个n x n的单位矩阵张量。
	   5. torch.arange(start, end, step)：创建一个从 start 到 end（不包括）的等差序列张量。
	   6. torch.linspace(start, end, steps)：创建一个从 start 到 end（包括）的均匀间隔的序列张量，其中 steps 是序列的长度。
	   7. `torch.rand(shape)`：创建指定形状的均匀分布随机值张量，取值范围为`[0, 1)。`
	   8. `torch.randn(shape)`：创建指定形状的正态分布随机值张量，均值为0，标准差为1。`
	   9. torch.empty(shape)`：创建指定形状的未初始化张量，张量的值将取决于内存中的内容。
	   10. torch.normal(mean,std)：正态分布(均值为mean，标准差是std)
2. 张量运算：
   - torch.add(tensor1, tensor2)：逐元素相加两个张量。
   - torch.sub(tensor1, tensor2)：逐元素相减两个张量。
   - torch.mul(tensor1, tensor2)：逐元素相乘两个张量。
   - torch.div(tensor1, tensor2)：逐元素相除两个张量。
   - torch.matmul(tensor1, tensor2)：矩阵相乘两个张量。
3. 张量变形：
   - tensor.view(shape)：改变张量的形状。
   - tensor.reshape(shape)：返回一个具有指定形状的新张量。
4. 张量索引和切片：
   - tensor[index]：通过索引获取张量中的元素。
   - tensor[start:end]：通过切片获取张量中的部分元素。
5. 张量统计：
   - tensor.mean()：计算张量的均值。
   - tensor.sum()：计算张量的总和。
   - tensor.max()：找到张量中的最大值。
   - tensor.min()：找到张量中的最小值。
