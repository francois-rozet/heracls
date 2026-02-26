"""Tests for the heracls module."""

import pytest

from cattrs.errors import ClassValidationError
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
    output: str | Path
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: AdamConfig | SGDConfig = heracls.field(
        choices={"adam": AdamConfig, "sgd": SGDConfig},
        default_factory=AdamConfig,
    )
    dataset: str = "mnist"
    data_splits: tuple[float, ...] = (0.8, 0.1)
    n_epochs: int = 1024
    n_steps_per_epoch: int = 256
    tasks: Sequence[str] = field(default_factory=list)
    magic: Mapping[str, float] = field(default_factory=dict)


def test_from_dict() -> None:
    data = {
        "output": "./out",
        "model": {"dropout": 0.1},
        "optimizer": {"momentum": 0.9},
        "data_splits": [0.7, 0.2],
    }
    cfg = heracls.from_dict(TrainConfig, data)

    assert cfg == TrainConfig(
        output="./out",
        model=ModelConfig(dropout=0.1),
        optimizer=SGDConfig(momentum=0.9),
        data_splits=(0.7, 0.2),
    )

    data = {
        "output": "./out",
        "optimizer": AdamConfig(),
    }
    cfg = heracls.from_dict(TrainConfig, data)

    assert cfg == TrainConfig(
        output="./out",
        optimizer=AdamConfig(),
    )


def test_from_dotlist() -> None:
    data = ["output=./out", "model.dropout=0.1", "optimizer.nesterov=yes", "data_splits=[0.7,0.2]"]
    cfg = heracls.from_dotlist(TrainConfig, data)

    assert cfg == TrainConfig(
        output="./out",
        model=ModelConfig(dropout=0.1),
        optimizer=SGDConfig(nesterov=True),
        data_splits=(0.7, 0.2),
    )


def test_from_dict_union_strict_match() -> None:
    data = {
        "output": "./out",
        "optimizer": {"learning_rate": 0.1},
    }

    with pytest.raises(ClassValidationError):
        heracls.from_dict(TrainConfig, data)


def test_ArgumentParser() -> None:
    parser = heracls.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run")
    parser.add_arguments(TrainConfig, dest="train", root=True)

    parser._finalize().print_help()

    args = parser.parse_args([
        "--output",
        "./out",
    ])

    assert args.train == TrainConfig(output="./out")

    args = parser.parse_args([
        "--output",
        "./out",
        "--model.dropout",
        "0.1",
        "--optimizer",
        "sgd",
        "--data_splits",
        "[0.7, 0.2]",
        "--magic",
        r'{"p": 3, "i": 0.14}',
    ])

    assert args.train == TrainConfig(
        output="./out",
        model=ModelConfig(dropout=0.1),
        optimizer=SGDConfig(),
        data_splits=(0.7, 0.2),
        magic={"p": 3, "i": 0.14},
    )

    dump = heracls.to_yaml(args.train)

    assert heracls.from_yaml(TrainConfig, dump) == args.train
