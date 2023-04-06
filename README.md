# nn_optimization
Optimization using overparameterization with neural networks

## Installation

Clone the repo and execute the following commands from the repository's root.

Install the `nn_optimization` package in development mode:
```
pip install -e .
```

## Optimizing analytical functions

```
python scripts/train.py --config-name config
```
where `config` is a config file in `config/` that specifies the optimization setup.

The config can be overwritten through the command line. For example, when wanting to use
direct rather than indirect optimization:

```
python scripts/train.py --config-name config model=parameter
```

## Visualizing loss functions

Example:
```
python scripts/viz_loss.py --function_name ackley --range 40 --one_dim
```

See `python scripts/viz_loss.py -h` for the available arguments.
 