# How to train your own models using timm? 
In this tutorial, we will be looking at the training script of `timm`. There are various features that `timm` has to offer and some of them have been listed below: 
1. Auto Augmentation [paper](https://arxiv.org/abs/1805.09501)
2. Augmix 
3. Distributed Training on multiple GPUs
4. Mixed precision training 
5. Auxiliary Batch Norm for AdvProp [paper](https://arxiv.org/abs/1911.09665)
6. Synchronized Batch Norm 
7. Mixup and Cutmix with an ability to switch between the two & also turn-off augmentation at a certain epoch 
`timm` also supports multiple optimizers & schedulers. In this tutorial we will be only be looking at the above 7 features and look at how you could utilize `timm` to use these features for your own experiments on a custom dataset. 
As part of this tutorial, we will first start out with a general introduction to the training script and look at the various key steps that occur inside this script at a high-level. Then, we will look at some of the details of the above 7 features to get a further understanding of `train.py`. 
## Training args
The training script in `timm` can accept ~100 arguments. You can find more about these by running `python train.py --help`. These arguments are to define Dataset/Model parameters, Optimizer parameters, Learnining Rate scheduler parameters, Augmentation and regularization, Batch Norm parameters, Model exponential moving average parameters, and some miscellaneaous parameters such as `--seed`, `--tta` etc.
As part of this tutorial, we will be looking at how the training script makes use of these parameters from a high-level view. This could be beneficial for you to able to run your own experiments on ImageNet or any other custom dataset using `timm`. 
### Required args
The only argument required by `timm` training script is the path to the training data such as ImageNet which is structured in the following way: 
```
imagenette2-320
├── train
│   ├── n01440764
│   ├── n02102040
│   ├── n02979186
│   ├── n03000684
│   ├── n03028079
│   ├── n03394916
│   ├── n03417042
│   ├── n03425413
│   ├── n03445777
│   └── n03888257
└── val
    ├── n01440764
    ├── n02102040
    ├── n02979186
    ├── n03000684
    ├── n03028079
    ├── n03394916
    ├── n03417042
    ├── n03425413
    ├── n03445777
    └── n03888257
```
So to start training on this `imagenette2-320` we could just do something like `python train.py <path_to_imagenette2-320_dataset>`.
### Default args
The various default args, in the training script are setup for you and what get's passed to the training script looks something like this: 
```python 
Namespace(aa=None, amp=False, apex_amp=False, aug_splits=0, batch_size=32, bn_eps=None, bn_momentum=None, bn_tf=False, channels_last=False, clip_grad=None, color_jitter=0.4, cooldown_epochs=10, crop_pct=None, cutmix=0.0, cutmix_minmax=None, data_dir='../imagenette2-320', dataset='', decay_epochs=30, decay_rate=0.1, dist_bn='', drop=0.0, drop_block=None, drop_connect=None, drop_path=None, epochs=200, eval_metric='top1', gp=None, hflip=0.5, img_size=None, initial_checkpoint='', input_size=None, interpolation='', jsd=False, local_rank=0, log_interval=50, lr=0.01, lr_cycle_limit=1, lr_cycle_mul=1.0, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, mean=None, min_lr=1e-05, mixup=0.0, mixup_mode='batch', mixup_off_epoch=0, mixup_prob=1.0, mixup_switch_prob=0.5, model='resnet101', model_ema=False, model_ema_decay=0.9998, model_ema_force_cpu=False, momentum=0.9, native_amp=False, no_aug=False, no_prefetcher=False, no_resume_opt=False, num_classes=None, opt='sgd', opt_betas=None, opt_eps=None, output='', patience_epochs=10, pin_mem=False, pretrained=False, ratio=[0.75, 1.3333333333333333], recount=1, recovery_interval=0, remode='const', reprob=0.0, resplit=False, resume='', save_images=False, scale=[0.08, 1.0], sched='step', seed=42, smoothing=0.1, split_bn=False, start_epoch=None, std=None, sync_bn=False, torchscript=False, train_interpolation='random', train_split='train', tta=0, use_multi_epochs_loader=False, val_split='validation', validation_batch_size_multiplier=1, vflip=0.0, warmup_epochs=3, warmup_lr=0.0001, weight_decay=0.0001, workers=4)
```
Notice, that `args` is a `Namespace` which means we can set more along the way if needed by doing something like `args.new_variable="some_value"`.
To get a one-line introduction of these various arguments, we can just do something like `python train.py --help`.
> NOTE: We will be looking at what these parameters in greater detail as part of this tutorial. Do note that some random augmentations are set by default such as `color_jitter`, `hfliip` but there is a parameter `no-aug` in case you wanted to turn of all training data augmentations. Also, the default optimizer `opt` is 'sgd' but it is possible to change that. `timm` offers a vast number of optimizers to train your models with. 
## The training script in 20 steps
In this section we will look at the various steps from a high level perspective that occur inside the training script. These steps have been outlined below in the correct order:
1. Setup up <u>distributed training parameters</u> if `args.distributed` is `True`. 
2. Setup <u>manual seed</u> for reproducible results. 
3. **Create Model**: Create the model to train using `timm.create_model` function. 
4. Setup <u>data config</u> based on model's default config. In general the default config of the model looks something like: 
```python
{'url': '', 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7), 'crop_pct': 0.875, 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'first_conv': 'conv1', 'classifier': 'fc'}
```
5. Setup <u>augmentation batch splits</u> and if the number of augmentation batch splits is more than 1, and if so, convert all model BatchNormlayers to Split Batch Normalization layers.
> NOTE: I feel this needs a little more explaination. In general, when we train a model, we apply the data augmentation to the complete batch and then define batch norm statistics from this complete batch such as mean and variance. But as introduced in this [paper](https://arxiv.org/abs/1911.09665), sometimes it is beneficial to split the data into groups and use separate Batch Normalization layers for each to normalize the groups independently throughout the training process. This is referred to as auxiliary batch norm in the paper and is referred to `SplitBatchNorm2d` in `timm`. 
> NOTE: Let's assume that number of augmentation batch splits is set to two. In that case, we split the data into two groups - one without any augmentations (referred to as clean) and one with augmentations. Then we use two separate batch normalization layers to normalize the two groups throughout the training process. 
6. If we are using multiple GPUs for training, then setup either apex syncBN or PyTorch native [SyncBatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html) to set up <u>Synchronized Batch Normalization</u>. This means that rather than normalizing the data on each individual GPU, we normalize the whole batch at one spread across multiple GPUs. 
7. Make model exportable using `torch.jit` if requested. 
8. <u>Initialize optimizer</u> based on arguments passed to the training script. 
9. Setup <u>mixed Precision</u> - either using `apex.amp` or using native torch amp - `torch.cuda.amp.autocast`. 
10. Load model weights if resuming from a <u>model checkpoint</u>.
11. Setup <u>exponential moving average</u> of model weights. This is similar to [Stochastic Weight Averaging](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/). 
12. Setup <u>distributed training</u> based on parameters from step-1.
13. Setup <u>learning rate scheduler</u>.
14. Create <u>training and validation dataset</u>.
15. Setup <u>Mixup/Cutmix</u> data augmentation.
16. Convert training dataset to <u>`AugmixDataset`</u> if number of augmentation batch splits from step-5 is greater than 1. 
17. Create <u>training data loader and Validation dataloader</u>.
> NOTE: Transforms/Augmentations also get created inside the training dataloader.
18. Setup <u>Loss</u> function. 
19. Setup <u>model checkpointing and evaluation metrics</u>. 
20. <u>Train and Validate</u> the model and also store the eval metrics to an output file.
## Some key `timm` features
### Auto-Augment
```python
#hide
# TODO: reference docs for auto-augment here
```
To enable auto augmentation during training - 
```python
python train.py ./imagenette2-320 --aa 'v0'
```
### Augmix
A brief introduction about augmix has been presented [here](https://fastai.github.io/timmdocs/dataset.html#AugmixDataset). To enable augmix during training, simply do: 
```python 
python train.py ./imagenette2-320 --aug-splits 3 --jsd 
```
`timm` also supports augmix with `RandAugment` and `AutoAugment` like so: 
```python 
python train.py ./imagenette2-320 --aug-splits 3 --jsd --aa rand-m9-mstd0.5-inc1
```
### Distributed Training on multiple GPUs
To train models on multiple GPUs, simply replace `python train.py` with `./distributed_train.sh <num-gpus>` like so:
```python
./distributed_train.sh 4 ./imagenette2-320 --aug-splits 3 --jsd 
```
This trains the model using `AugMix` data augmentation on 4 GPUs.
### Mixed precision training
To enable mixed precision training, simply add the `--amp` flag. `timm` will automatically implement mixed precision training either using `apex` or PyTorch Native mixed precision training. 
```python
python train.py ../imagenette2-320 --aug-splits 3 --jsd --amp 
```
### Auxiliary Batch Norm/ `SplitBatchNorm`
```python
#hide
# TODO: refer blog post or tutorial here
```
From the paper, 
```markdown
Batch normalization serves as an essential component for many state-of-the-art computer vision models. Specifically, BN normalizes input features by the mean and variance computed within each mini-batch. **One intrinsic assumption of utilizing BN is that the input features should come from a single or similar distributions.** This normalization behavior could be problematic if the mini-batch contains data from different distributions, there- fore resulting in inaccurate statistics estimation.
To disentangle this mixture distribution into two simpler ones respectively for the clean and adversarial images, we hereby propose an auxiliary BN to guarantee its normalization statistics are exclusively preformed on the adversarial examples.
```
To enable split batch norm, 
```python 
python train.py ./imagenette2-320 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --split-bn
```
Using the above command, `timm` now has separate batch normalization layer for each augmentation split. 
### Synchronized Batch Norm
Synchronized batch norm is only used when training on multiple GPUs. From [papers with code](https://paperswithcode.com/method/syncbn):
```markdown
Synchronized Batch Normalization (SyncBN) is a type of batch normalization used for multi-GPU training. Standard batch normalization only normalizes the data within each device (GPU). SyncBN normalizes the input within the whole mini-batch.
```
To enable, simply add `--sync-bn` flag like so: 
```python
./distributed_train.sh 4 ../imagenette2-320 --aug-splits 3 --jsd --sync-bn  
```
### Mixup and Cutmix
To enable either mixup or cutmix, simply add the `--mixup` or `--cutmix` flag with alpha value.  
Default probability of applying the augmentation is 1.0. If you need to change it, use `--mixup-prob` argument with new value.
For example, to enable mixup, 
```python 
train.py ../imagenette2-320 --mixup 0.5
train.py ../imagenette2-320 --mixup 0.5 --mixup-prob 0.7
```
Or for Cutmix, 
```python 
train.py ../imagenette2-320 --cutmix 0.5
train.py ../imagenette2-320 --cutmix 0.5 --mixup-prob 0.7
```
It is also possible to enable both, 
```python 
python train.py ../imagenette2-320 --mixup 0.5 --cutmix 0.5 --mixup-switch-prob 0.3
```
The above command will use either Mixup or Cutmix as data augmentation techniques and apply it to the batch with 50% probability. It will also switch between the two with 30% probability (Mixup - 70%, 30% switch to Cutmix).
There is also a parameter to turn off Mixup/Cutmix augmentation at a certail epoch:
```python 
python train.py ../imagenette2-320 --mixup 0.5 --cutmix 0.5 --mixup-switch-prob 0.3 --mixup-off-epoch 10
```
The above command only applies the Mixup/Cutmix data augmentation for the first 10 epochs. 
