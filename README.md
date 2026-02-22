# Heracls - Slayer of Hydra

`heracls` is a tiny utility package to instantiate typed dataclasses from flexible config sources (dictionary, OmegaConf, YAML, dotlist, CLI, ...). It is designed for projects that want strict, typed config objects while supporting the dynamic overrides commonly used in scripts and experiments.

## Installation

The `heracls` package is available on [PyPi](https://pypi.org/project/heracls) and can be installed with `pip`.

```
pip install heracls
```

Alternatively, if you need the latest features, you can install it from source.

```
pip install git+https://github.com/francois-rozet/heracls
```

## Getting started

The following example demonstrates how to declare a nested dataclass config, instantiate it from command line arguments, serialize it to YAML, and use it in a script. The `heracls.ArgumentParser` is a thin wrapper around [simple-parsing](https://github.com/lebrice/SimpleParsing), which populates the parser arguments from the config structure and types.

```python
import heracls

from dataclasses import dataclass, field
from typing import Literal, Tuple, Union

@dataclass
class ModelConfig:
    name: str = "mlp"
    depth: int = 3
    width: int = 256

@dataclass
class AdamConfig:
    name: Literal["adam"] = "adam"
    betas: Tuple[float, float] = (0.95, 0.95)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class SGDConfig:
    name: Literal["sgd"] = "sgd"
    momentum: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: Union[AdamConfig, SGDConfig] = heracls.choice(
        {"adam": AdamConfig, "sgd": SGDConfig},
        default="adam",
    )
    dataset: str = "mnist"
    data_splits: Tuple[float, float] = (0.8, 0.1)
    n_epochs: int = 1024
    n_steps_per_epoch: int = 256

def main():
    parser = heracls.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run")
    parser.add_arguments(TrainConfig, "train")

    args = parser.parse_args()

    print(heracls.to_yaml(args.train))

    if args.dry:
        return

    trainset, validset, testset = load_dataset(args.train.data_splits)

    model = init_model(args.train.model)

    for epoch in range(args.train.n_epochs):
        ...

if __name__ == "__main__":
    main()
```

```
$ python train.py --dry --depth 5 --optimizer sgd --data_splits 0.7 0.2
model:
  name: mlp
  depth: 5
  width: 256
optimizer:
  name: sgd
  momentum: 0.0
  learning_rate: 0.001
  weight_decay: 0.0
dataset: mnist
data_splits:
- 0.7
- 0.2
n_epochs: 1024
n_steps_per_epoch: 256
```

```
$ python train.py --help
usage: train.py [-h] [--dry] [--optimizer {adam,sgd}] [--dataset str]
                [--data_splits float float] [--n_epochs int] [--n_steps_per_epoch int]
                [--model.name str] [--depth int] [--width int] [--optimizer.name {adam}]
                [--betas float float] [--learning_rate float] [--weight_decay float]

options:
  -h, --help                 show this help message and exit
  --dry                      dry run (default: False)

TrainConfig ['train']:
  TrainConfig(model: __main__.ModelConfig = <factory>, optimizer: Union[__main__.AdamConfig, __main__.SGDConfig] = <factory>, dataset: str = 'mnist', data_splits: Tuple[float, float] = (0.8, 0.1), n_epochs: int = 1024, n_steps_per_epoch: int = 256)

  --optimizer {adam,sgd}     (default: adam)
  --dataset str              (default: mnist)
  --data_splits float float  (default: (0.8, 0.1))
  --n_epochs int             (default: 1024)
  --n_steps_per_epoch int    (default: 256)

ModelConfig ['train.model']:
  ModelConfig(name: str = 'mlp', depth: int = 3, width: int = 256)

  --model.name str           (default: mlp)
  --depth int                (default: 3)
  --width int                (default: 256)

AdamConfig ['train.optimizer']:
  AdamConfig(name: Literal['adam'] = 'adam', betas: Tuple[float, float] = (0.95, 0.95), learning_rate: float = 0.001, weight_decay: float = 0.0)

  --optimizer.name {adam}    (default: adam)
  --betas float float        (default: (0.95, 0.95))
  --learning_rate float      (default: 0.001)
  --weight_decay float       (default: 0.0)
```
