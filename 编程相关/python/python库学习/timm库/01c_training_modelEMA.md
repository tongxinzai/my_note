# Model EMA (Exponential Moving Average)
When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.
> NOTE: A smoothed version of the weights is necessary for some training schemes to perform well. Example Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA smoothing of weights to match results.
`timm` supports EMA similar to [tensorflow](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage).
To train models with EMA simply add the `--model-ema` flag and `--model-ema-decay` flag with a value to define the decay rate for EMA. 
To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but disable validation of the EMA weights. Validation will have to be done manually in a separate process, or after the training stops converging.
> NOTE: This class is sensitive where it is initialized in the sequence of model init, GPU assignment and distributed training wrappers.
## Training without EMA 
```python
python train.py ../imagenette2-320 --model resnet34
```
## Training with EMA 
```python
python train.py ../imagenette2-320 --model resnet34 --model-ema --model-ema-decay 0.99
```
The above training script means that when updating the model weights, we keep 99.99% of the previous model weights and only update 0.01% of the new weights at each iteration. 
```python"
model_weights = decay * model_weights + (1 - decay) * new_model_weights
```
### Internals of Model EMA inside `timm`
Inside `timm`, when we pass `--model-ema` flag then `timm` wraps the model class inside `ModelEmaV2` class which looks like:
```python 
class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
```
Basically, we initialize the `ModeEmaV2` by passing in an existing `model` and a decay rate, in this case `decay=0.9999`. 
This looks something like `model_ema = ModelEmaV2(model)`. Here, `model` could be any existing model as long as it's created using the `timm.create_model` function. 
Next, during training especially inside the `train_one_epoch`, we call the `update` method of `model_ema` like so: 
```python
if model_ema is not None:
    model_ema.update(model)
```
All parameter updates based on `loss` occur for `model`. When we call `optimizer.step()`, then the `model` weights get updated and not the `model_ema`'s weights. 
Therefore, when we call the `model_ema.update` method, as can be seen, this calls the `_update` method with `update_fn = lambda e, m: self.decay * e + (1. - self.decay) * m)`. 
> NOTE: Basically, here, `e` refers to `model_ema` and `m` refers to the `model` whose weights get updated during training.  The `update_fn` specifies that we keep `self.decay` times the `model_ema` and `1-self.decay` times the `model`. 
Thus when we call the `_update` function it goes through each of the parameters inside `model` and `model_ema` and updates the state for `model_ema` to keep 99.99% of the existing state and 0.01% of the new state. 
> NOTE: Note that `model` and `model_ema` have the same keys inside the `state_dict`.
