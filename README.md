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

The following example demonstrates how to declare a nested dataclass config, instantiate it from command line arguments, serialize it to YAML, and use it in a script. The `heracls.ArgumentParser` parser infers arguments from the dataclass structure and types.

```python
from dataclasses import dataclass, field
from typing import Literal

import heracls

@dataclass
class ModelConfig:
    name: str = "mlp"
    depth: int = 3
    width: int = 256
    dropout: float | None = None

@dataclass
class AdamConfig:
    name: Literal["adam"] = "adam"
    betas: tuple[float, float] = (0.95, 0.95)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class SGDConfig:
    name: Literal["sgd"] = "sgd"
    momentum: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    nesterov: bool = False

@dataclass
class TrainConfig:
    output: str
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: AdamConfig | SGDConfig = heracls.choice(
        {"adam": AdamConfig, "sgd": SGDConfig},
        default="adam",
    )
    dataset: str = "mnist"
    data_splits: tuple[float, ...] = (0.8, 0.1)
    n_epochs: int = 1024
    n_steps_per_epoch: int = 256
    tasks: list[str] = field(default_factory=list)

def main() -> None:
    parser = heracls.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run")
    parser.add_arguments(TrainConfig, dest="train", root=True)

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
$ python examples/train.py --dry --output ./out --model.dropout 0.1 --optimizer sgd --data_splits '[0.7, 0.2]'
output: ./out
model:
  name: mlp
  depth: 3
  width: 256
  dropout: 0.1
optimizer:
  name: sgd
  momentum: 0.0
  learning_rate: 0.001
  weight_decay: 0.0
  nesterov: false
dataset: mnist
data_splits:
- 0.7
- 0.2
n_epochs: 1024
n_steps_per_epoch: 256
tasks: []
```

```
$ python examples/train.py --help
usage: train.py [-h] [--dry] --output str [--model.name str] [--model.depth int]
                [--model.width int] [--model.dropout float] [--optimizer {adam,sgd}]
                [--dataset str] [--data_splits tuple[float, ...]] [--n_epochs int]
                [--n_steps_per_epoch int] [--tasks list[str]] [--optimizer.name {adam}]
                [--optimizer.betas tuple[float, float]] [--optimizer.learning_rate float]
                [--optimizer.weight_decay float]

options:
  -h, --help                             show this help message and exit
  --dry                                  dry run (default: False)

  --output str
  --optimizer {adam,sgd}                 (default: adam)
  --dataset str                          (default: mnist)
  --data_splits tuple[float, ...]        (default: (0.8, 0.1))
  --n_epochs int                         (default: 1024)
  --n_steps_per_epoch int                (default: 256)
  --tasks list[str]                      (default: [])

model:
  --model.name str                       (default: mlp)
  --model.depth int                      (default: 3)
  --model.width int                      (default: 256)
  --model.dropout float                  (default: None)

optimizer:
  --optimizer.name {adam}                (default: adam)
  --optimizer.betas tuple[float, float]  (default: (0.95, 0.95))
  --optimizer.learning_rate float        (default: 0.001)
  --optimizer.weight_decay float         (default: 0.0)
```
