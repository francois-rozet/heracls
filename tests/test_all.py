"""Tests for the heracls module."""

from dataclasses import dataclass, field
from typing import Literal, Tuple, Union

from heracls import ArgumentParser, choice, from_yaml, to_yaml


def test_pipeline():
    @dataclass
    class ModelConfig:
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
        optimizer: Union[AdamConfig, SGDConfig] = choice(
            {"adam": AdamConfig, "sgd": SGDConfig},
            default="adam",
        )
        dataset: str = "mnist"
        data_splits: Tuple[float, float] = (0.8, 0.1)
        n_epochs: int = 1024
        n_steps_per_epoch: int = 256

    parser = ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run")
    parser.add_arguments(TrainConfig, "train")

    args = parser.parse_args(["--depth", "5", "--optimizer", "sgd", "--data_splits", "0.7", "0.2"])

    assert args.train == TrainConfig(
        model=ModelConfig(depth=5),
        optimizer=SGDConfig(),
        data_splits=(0.7, 0.2),
    )

    dump = to_yaml(args.train)

    assert from_yaml(TrainConfig, dump) == args.train
