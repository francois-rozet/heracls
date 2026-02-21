"""Tests for the heracls.transforms module."""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple, Union

from heracls.transforms import from_dotlist, from_yaml, to_yaml


def test_dl_config():
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

    dotlist = ["optim.optimizer=sgd", "data_splits=[0.7,0.2]", "slurm.account=frozet"]
    cfg = from_dotlist(TrainConfig, dotlist)

    assert cfg == TrainConfig(
        optim=SGDConfig(),
        data_splits=(0.7, 0.2),
        slurm={"account": "frozet"},
    )

    dump = to_yaml(cfg)

    assert from_yaml(TrainConfig, dump) == cfg
