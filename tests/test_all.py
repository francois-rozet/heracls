"""Tests for the heracls module."""

from dataclasses import dataclass, field
from typing import Literal, Union

from heracls import ArgumentParser, choice, from_yaml, to_yaml


def test_pipeline():
    @dataclass
    class ModelConfig:
        name: str = "mlp"
        depth: int = 3
        width: int = 256

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
        model: ModelConfig = field(default_factory=ModelConfig)
        optimizer: Union[AdamConfig, SGDConfig] = choice(
            {"adam": AdamConfig, "sgd": SGDConfig},
            default="adam",
        )
        dataset: str = "mnist"
        data_splits: tuple[float, ...] = (0.8, 0.1)
        n_epochs: int = 1024
        n_steps_per_epoch: int = 256
        tasks: list[str] = field(default_factory=list)

    parser = ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run")
    parser.add_arguments(TrainConfig, dest="train", root=True)

    args = parser.parse_args([
        "--model.depth",
        "5",
        "--optimizer",
        "sgd",
        "--data_splits",
        "0.7",
        "0.2",
    ])

    assert args.train == TrainConfig(
        model=ModelConfig(depth=5),
        optimizer=SGDConfig(),
        data_splits=(0.7, 0.2),
    )

    dump = to_yaml(args.train)

    assert from_yaml(TrainConfig, dump) == args.train
