```python
#hide
from nbdev.showdoc import *
```
# Loss Functions
```python
import timm
import torch
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import mixup_target
```
## LabelSmoothingCrossEntropy
Same as NLL loss with label smoothing. Label smoothing increases loss when the model is correct `x` and decreases loss when model is incorrect `x_i`. Use this to not punish model as harshly, such as when incorrect labels are expected. 
```python
x = torch.eye(2)
x_i = 1 - x
y = torch.arange(2)
```
```python
LabelSmoothingCrossEntropy(0.0)(x,y),LabelSmoothingCrossEntropy(0.0)(x_i,y)
```
    (tensor(0.3133), tensor(1.3133))
```python
LabelSmoothingCrossEntropy(0.1)(x,y),LabelSmoothingCrossEntropy(0.1)(x_i,y)
```
    (tensor(0.3633), tensor(1.2633))
## SoftTargetCrossEntropy
`log_softmax` family loss function to be used with mixup.  Use __[mixup_target](https://github.com/rwightman/pytorch-image-models/blob/9a38416fbdfd0d38e6922eee5d664e8ec7fbc356/timm/data/mixup.py#L22)__ to add label smoothing and adjust the amount of mixing of the target labels. 
```python
x=torch.tensor([[[0,1.,0,0,1.]],[[1.,1.,1.,1.,1.]]],device='cuda')
y=mixup_target(torch.tensor([1,4],device='cuda'),5, lam=0.7)
x,y
```
    (tensor([[[0., 1., 0., 0., 1.]],
             [[1., 1., 1., 1., 1.]]], device='cuda:0'),
     tensor([[0.0000, 0.7000, 0.0000, 0.0000, 0.3000],
             [0.0000, 0.3000, 0.0000, 0.0000, 0.7000]], device='cuda:0'))
```python
SoftTargetCrossEntropy()(x[0],y),SoftTargetCrossEntropy()(x[1],y)
```
    (tensor(1.1326, device='cuda:0'), tensor(1.6094, device='cuda:0'))
```python
#hide
from nbdev.export import notebook2script
notebook2script()
```
    Converted 00_core.ipynb.
    Converted 01_training.ipynb.
    Converted 19_loss.cross_entropy.ipynb.
    Converted index.ipynb.
    
