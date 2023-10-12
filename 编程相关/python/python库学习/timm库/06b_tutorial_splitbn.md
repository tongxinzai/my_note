# What is Split Batch Normalization and how can we implement it? 
Split Batch Normalization was first introduced in [Split Batch Normalization: Improving Semi-Supervised Learning under Domain Shift](https://arxiv.org/abs/1904.03515).
From the abstract of the paper: 
    
```
Recent work has shown that using unlabeled data in semisupervised learning is not always beneficial and can even hurt generalization, especially when there is a class mismatch between the unlabeled and labeled examples. We investigate this phenomenon for image classification on the CIFAR-10 and the ImageNet datasets, and with many other forms of domain shifts applied (e.g. salt-and-pepper noise). Our main contribution is Split Batch Normalization (Split-BN), a technique to improve SSL when the additional unlabeled data comes from a shifted distribution. We achieve it by using separate batch normalization statistics for unlabeled examples. Due to its simplicity, we recommend it as a standard practice. Finally, we analyse how domain shift affects the SSL training process. In particular, we find that during training the statistics of hidden activations in late layers become markedly different between the unlabeled and the labeled examples.
```
In simple words, they propose to compute separately batch normalization statistics for the unsupervised and supervised dataset. That is, have separate BN layers instead of 1 for the whole batch. 
You might say that's easy to say but how do we implement in code? 
Well, in `timm` training, you just do: 
```
python train.py ../imagenette2-320 --aug-splits 3 --split-bn --aa rand-m9-mstd0.5-inc1 --resplit 
```
And that's it. But what does this command mean? 
Running the above command- 
1. Creates 3 groups of training batches 
    1. The first one is referred to as the original (with minimal or zero augmentation)
    2. The second one is with random augmentation applied to the first one.
    3. The third one is again with random augmentation applied to the first one. 
    > NOTE: Random augmentations are stochastic. Therefore, the second and the third batch are different from each other. 
2. Converts every Batch Normalization inside the model to Split Batch Normalization Layer. 
3. Does not apply random erase to the first batch, also referred to as the first augmentation split. 
## `SplitBatchNorm2d`
The `SplitBatchNorm2d` on it's own is few lines of code: 
```python 
class SplitBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_splits - 1)])
    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            return super().forward(input)
```
Basically, inside the [Adversarial Examples Improve Image Recognition](https://arxiv.org/abs/1911.09665) paper, the authors refer to this Split Batch Norm as Auxilary batch norm. Therefore, as we can see in code, `self.aux_bn` is a list of `num_splits-1` length.
Basically, because we subclass `torch.nn.BatchNorm2d`, therefore, this SplitBatchNorm2d is in itself an instance of Batch Normalization, therefore the first batch norm layer is the `nn.BatchNorm2d` itself which can be used to normalize the first augmentation split or the clean batch. 
Then, we create `num_splits-1` number of auxiliary batch norms to normalize the remaining splits in the input batch. 
This way, we normalize the input batch `X` separately depending on the number of splits. This is achieved inside these lines of code: 
```python 
split_input = input.split(split_size)
x = [super().forward(split_input[0])]
for i, a in enumerate(self.aux_bn):
    x.append(a(split_input[i + 1]))
return torch.cat(x, dim=0)
```
And that's how `timm` implements `SplitBatchNorm2d` in PyTorch :) 
