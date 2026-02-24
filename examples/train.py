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

    # trainset, validset, testset = load_dataset(args.train.data_splits)
    # model = init_model(args.train.model)

    # for epoch in range(args.train.n_epochs):
    #     ...


if __name__ == "__main__":
    main()
