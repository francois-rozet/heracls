"""Tests for the heracls module."""

from dataclasses import dataclass, field
from typing import Literal

from heracls import ArgumentParser, choice, from_yaml, to_yaml


def test_pipeline() -> None:
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
        model: ModelConfig = field(default_factory=ModelConfig)
        optimizer: AdamConfig | SGDConfig = choice(
            {"adam": AdamConfig, "sgd": SGDConfig},
            default="adam",
        )
        dataset: str = "mnist"
        data_splits: tuple[float, ...] = (0.8, 0.1)
        n_epochs: int = 1024
        n_steps_per_epoch: int = 256
        tasks: list[str] = field(default_factory=list)
        magic: dict[str, float] = field(default_factory=dict)

    parser = ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run")
    parser.add_arguments(TrainConfig, dest="train", root=True)

    parser.print_help()

    args = parser.parse_args([
        "--model.dropout",
        "0.1",
        "--optimizer",
        "sgd",
        "--data_splits",
        "0.7",
        "0.2",
        "--magic",
        r'{"p": 3, "i": 0.14}',
    ])

    assert args.train == TrainConfig(
        model=ModelConfig(dropout=0.1),
        optimizer=SGDConfig(),
        data_splits=(0.7, 0.2),
        magic={"p": 3, "i": 0.14},
    )

    dump = to_yaml(args.train)

    assert from_yaml(TrainConfig, dump) == args.train
