# Heracls - Slayer of Hydra

`heracls` is a tiny utility package to instantiate typed dataclasses from flexible config sources (dictionary, OmegaConf, YAML, dotlist, ...). It is designed for projects that want strict, typed config objects while supporting the dynamic overrides commonly used in scripts and experiments.

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

The following example demonstrates how to declare a nested dataclass config, instantiate it while overriding default fields with a dotlist, serialize it to YAML, and use its fields in a script.

```python
import heracls

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple, Union

@dataclass
class AdamConfig:
    optimizer: Literal["adam", "adamw"] = "adam"
    betas: Tuple[float, float] = (0.95, 0.95)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class SGDConfig:
    optimizer: Literal["sgd"] = "sgd"
    momentum: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    nesterov: bool = False

@dataclass
class TrainConfig:
    optim: Union[AdamConfig, SGDConfig] = field(default_factory=AdamConfig)
    n_epochs: int = 1024
    data_splits: Tuple[float, float] = (0.8, 0.1)
    slurm: Dict[str, Any] = field(default_factory=dict)

dotlist = [  # usually retrieved from the command line
    "optim.optimizer=sgd",
    "data_splits=[0.7,0.2]",
    "slurm.account=frozet",
]

cfg = heracls.from_dotlist(TrainConfig, dotlist)

with open("config.yaml", "w") as f:
    f.write(heracls.to_yaml(cfg))

trainset, validset, testset = load_dataset(cfg.data_splits)

for epoch in range(cfg.n_epochs):
    ...
```

The fields that are not specified in the dotlist are instantiated with their default value, as can be seen in the dumped `config.yaml` file.

```yaml
optim:
  optimizer: sgd
  momentum: 0.0
  learning_rate: 0.001
  weight_decay: 0.0
  nesterov: false
n_epochs: 1024
data_splits:
- 0.7
- 0.2
slurm:
  account: frozet
```
