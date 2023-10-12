# PlateauLRScheduler
In this tutorial we are going to be looking at the `PlateauLRScheduler` in the `timm` library.
```python
#hide
%load_ext autoreload
%autoreload 2
```
```python
from timm.scheduler.plateau_lr import PlateauLRScheduler
from nbdev.showdoc import show_doc
```
```python
show_doc(PlateauLRScheduler)
```
<h2 id="PlateauLRScheduler" class="doc_header"><code>class</code> <code>PlateauLRScheduler</code><a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm/scheduler/plateau_lr.py#L12" class="source_link" style="float:right">[source]</a></h2>
> <code>PlateauLRScheduler</code>(**`optimizer`**, **`decay_rate`**=*`0.1`*, **`patience_t`**=*`10`*, **`verbose`**=*`True`*, **`threshold`**=*`0.0001`*, **`cooldown_t`**=*`0`*, **`warmup_t`**=*`0`*, **`warmup_lr_init`**=*`0`*, **`lr_min`**=*`0`*, **`mode`**=*`'max'`*, **`noise_range_t`**=*`None`*, **`noise_type`**=*`'normal'`*, **`noise_pct`**=*`0.67`*, **`noise_std`**=*`1.0`*, **`noise_seed`**=*`None`*, **`initialize`**=*`True`*) :: `Scheduler`
Decay the LR by a factor every time the validation loss plateaus.
The `PlateauLRScheduler` as shown above accepts an `optimizer` and also some hyperparams which we will look into in detail below. We will first see how we can train models using the `PlateauLRScheduler` by first using `timm` training docs and then look at how we can use this scheduler as standalone scheduler for our custom training scripts. 
## Using `PlateauLRScheduler` scheduler with `timm` training script
To train models using the `PlateauLRScheduler` we simply update the training script args passed by passing in `--sched plateau` parameter alongside the necessary hyperparams. In this section we will also look at how each of the hyperparams update the `plateau` scheduler. 
The training command to use `cosine` scheduler looks something like: 
```python 
python train.py ../imagenette2-320/ --sched plateau
```
The `PlateauLRScheduler` by default tracks the `eval-metric` which is by default `top-1` in the `timm` training script. If the performance plateaus, then the new learning learning after a certain number of epochs (by default 10) is set to `lr * decay_rate`. This scheduler underneath uses PyTorch's [ReduceLROnPlateau](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau). 
## Args
All arguments passed to this scheduler are the same as PyTorch's `ReduceLROnPlateau` except they are renamed as follows: 
| TIMM      | PyTorch |
| ----------- | ----------- |
| patience_t      | patience       |
| decay_rate   | factor        |
| verbose      | verbose       |
| threshold   | threshold        |
| cooldown_t   | cooldown        |
| mode   | mode        |
| lr_min   | min_lr        |
The functionality is very similar to [ReduceLROnPlateau](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau) except the addition of Noise.
