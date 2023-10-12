# What all goes on inside the create_model function?
In this tutorial, we will be taking a deep dive inside the source code of the `create_model` function. We will also how can we convert any given into a feature extractor. We have already seen an example of this [here](https://fastai.github.io/timmdocs/create_model#Turn-any-model-into-a-feature-extractor). We converted a `ResNet-34` architecture to a feature extractor to extract features from the 2nd, 3rd and 4th layers. 
In this tutorial we are going to dig deeper into the `create_model` source code and have a look at how is `timm` able to convert any model to a feature extractor. 
## The `create_model` function
The `create_model` function is what is used to create hundreds of models inside `timm`. It also expects a bunch of `**kwargs` such as `features_only` and `out_indices` and passing these two `**kwargs` to the `create_model` function creates a feature extractor instead. Let's see how?
The `create_model` function itself is only around 50-lines of code. So all the magic has to happen somewhere else. As you might already know, every model name inside `timm.list_models()` is actually a function. 
As an example:
```python
%load_ext autoreload
%autoreload 2
```
```python
import timm
import random 
from timm.models import registry
m = timm.list_models()[-1]
registry.is_model(m)
```
    True
`timm` has an internal dictionary called `_model_entrypoints` that contains all the model names and their respective constructor functions. As an example, we could see get the constructor function for our `xception71` model through the `model_entrypoint` function inside `_model_entrypoints`.
```python
constuctor_fn = registry.model_entrypoint(m)
constuctor_fn
```
    <function timm.models.xception_aligned.xception71(pretrained=False, **kwargs)>
As we can see there is a function called `xception71` inside `timm.models.xception_aligned` module. Similarly, every model has a constructor function inside `timm`. In fact, this internal `_model_entrypoints` dictionary looks something like: 
```python
_model_entrypoints
>> 
{
'cspresnet50': <function timm.models.cspnet.cspresnet50(pretrained=False, **kwargs)>,
'cspresnet50d': <function timm.models.cspnet.cspresnet50d(pretrained=False, **kwargs)>,
'cspresnet50w': <function timm.models.cspnet.cspresnet50w(pretrained=False, **kwargs)>,
'cspresnext50': <function timm.models.cspnet.cspresnext50(pretrained=False, **kwargs)>,
'cspresnext50_iabn': <function timm.models.cspnet.cspresnext50_iabn(pretrained=False, **kwargs)>,
'cspdarknet53': <function timm.models.cspnet.cspdarknet53(pretrained=False, **kwargs)>,
'cspdarknet53_iabn': <function timm.models.cspnet.cspdarknet53_iabn(pretrained=False, **kwargs)>,
'darknet53': <function timm.models.cspnet.darknet53(pretrained=False, **kwargs)>,
'densenet121': <function timm.models.densenet.densenet121(pretrained=False, **kwargs)>,
'densenetblur121d': <function timm.models.densenet.densenetblur121d(pretrained=False, **kwargs)>,
'densenet121d': <function timm.models.densenet.densenet121d(pretrained=False, **kwargs)>,
'densenet169': <function timm.models.densenet.densenet169(pretrained=False, **kwargs)>,
'densenet201': <function timm.models.densenet.densenet201(pretrained=False, **kwargs)>,
'densenet161': <function timm.models.densenet.densenet161(pretrained=False, **kwargs)>,
'densenet264': <function timm.models.densenet.densenet264(pretrained=False, **kwargs)>,
}
```
So, every model inside `timm` has a constructor defined inside the respective modules. For example, all ResNets have been defined inside `timm.models.resnet` module. Thus, there are two ways to create a `resnet34` model:
```python
import timm
from timm.models.resnet import resnet34
# using `create_model`
m = timm.create_model('resnet34')
# directly calling the constructor fn
m = resnet34()
```
In `timm`, you never really want to directly call the constructor function. All models should be created using the `create_model` function itself.
### Register model
The source code of the `resnet34` constructor function looks like: 
```python
@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet34', pretrained, **model_args)
```
> NOTE: You will find that every model inside `timm` has a `register_model` decorator. At the beginning, the `_model_entrypoints` is an empty dictionary. It is the `register_model` decorator that adds the given model function constructor along with it's name to `_model_entrypoints`. 
```python
def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''
    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn
```
As can be seen above, the `register_model` function does some pretty basic steps. But the main one that I'd like to highlight is this one 
```python
_model_entrypoints[model_name] = fn
```
Thus, it adds the given `fn` to `_model_entrypoints` where the key is `fn.__name__`. 
> NOTE: Can you now guess what does having `@register_model` decorator on the `resnet34` function do? It creates an entry inside the `_model_entrypoints` that looks like `{'resnet34': <function timm.models.resnet.resnet34(pretrained=False, **kwargs)>}`.
Also, just by looking at the source code of this `resnet34` constructor function, we can see that after setting up some `model_args` it then calls `create_resnet` function. Let's see how that looks like:
```python
def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)
```
So the `_create_resnet` function instead calls the `build_model_with_cfg` function passing in a constructor class `ResNet`, variant name `resnet34`, a `default_cfg` and some `**kwargs`. 
### Default config
Every model inside `timm` has a default config. This contains the URL for the model pretrained weights, the number of classes to classify, input image size, pooling size and so on. 
The default config of `resnet34` looks like: 
```python
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
'num_classes': 1000,
'input_size': (3, 224, 224),
'pool_size': (7, 7),
'crop_pct': 0.875,
'interpolation': 'bilinear',
'mean': (0.485, 0.456, 0.406),
'std': (0.229, 0.224, 0.225),
'first_conv': 'conv1',
'classifier': 'fc'}
```
This default config get's passed to the `build_model_with_cfg` function along side the other arguments such as the constructor class and some model arguments. 
### Build model with config
This `build_model_with_cfg` function is what's responsible for: 
1. Actually instantiating the model class to create the model inside `timm`
2. Pruning the model if `pruned=True` 
3. Loading the pretrained weights if `pretrained=True` 
4. Converting the model to a feature extractor if `features=True`
After inspecting the source code for this function:
```python
def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: dict = None,
        feature_cfg: dict = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Callable = None,
        pretrained_custom_load: bool = False,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = deepcopy(default_cfg)
    
    if pruned:
        model = adapt_model_from_file(model, variant)
    # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model)
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn, strict=pretrained_strict)
    
    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg
    
    return model
```
One can see that the model get's created at this point `model = model_cls(**kwargs)`. 
Also, as part of this tutorial we are not going to look inside `pruned` and `adapt_model_from_file` function.  
We have already understood and looked inside the `load_pretrained` function [here](https://fastai.github.io/timmdocs/models#My-dataset-doesn't-consist-of-3-channel-images---what-now?).
And we take a deep dive inside the `FeatureListNet` [here]() that is responsible for converting our deep learning model to a Feature Extractor. 
### Summary
That's really it. We have now completely looked at `timm.create_model` function. The main functions that get called are: 
- The model constructor function with is different for each model and set's up model specific arguments. The `_model_entrypoints` dictionary contains all the model names and respective constructor functions. 
- `build_with_model_cfg` function with accepts a model constructor class alongside the model specific arguments set inside the model constructor function.
- `load_pretrained` which loads the pretrained weights. This also works when the number of input channels is not equal to 3 as in the case of ImageNet. 
- `FeatureListNet` class that is responsible for converting any model into a feature extractor. 
