"""Data transformations."""

__all__ = [
    "from_dict",
    "from_dotlist",
    "from_omega",
    "from_yaml",
    "to_dict",
    "to_omega",
    "to_yaml",
]

import dacite

from dataclasses import fields as iter_fields
from dataclasses import is_dataclass
from omegaconf import DictConfig, OmegaConf
from typing import Any, ClassVar, Dict, List, Protocol, Type, TypeVar

T = TypeVar("T")


class DataClass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


def from_dict(data_cls: Type[T], data: Dict[str, Any]) -> T:
    """Instantiate a dataclass from a dictionary.

    Arguments:
        data_cls: A dataclass type.
        data: A dictionary.

    Returns:
        A `data_cls` instance.
    """
    return dacite.from_dict(
        data_class=data_cls,
        data=data,
        config=dacite.Config(
            cast=[tuple],
            strict=True,
            strict_unions_match=True,
        ),
    )


def to_dict(data: DataClass, /, recursive: bool = False) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary.

    Arguments:
        data: A dataclass instance.
        recursive: Recursively convert nested dataclasses when `True`.

    Returns:
        A dictionary representation of `data`.
    """
    if not is_dataclass(type(data)):
        return data
    keys = [f.name for f in iter_fields(data)]
    data = {k: getattr(data, k) for k in keys}
    if recursive:
        data = {k: to_dict(v, recursive=True) for k, v in data.items()}
    return data


def from_omega(data_cls: Type[T], data: DictConfig) -> T:
    """Instantiate a dataclass from an :mod:`omegaconf` config.

    Arguments:
        data_cls: A dataclass type.
        data: A config object.

    Returns:
        A `data_cls` instance.
    """
    return from_dict(
        data_cls,
        OmegaConf.to_container(data, resolve=True, throw_on_missing=True),
    )


def to_omega(data: DataClass) -> DictConfig:
    """Convert a dataclass instance to an :mod:`omegaconf` config.

    Arguments:
        data: A dataclass instance.

    Returns:
        A config representation of `data`.
    """
    return OmegaConf.create(to_dict(data, recursive=True))


def from_yaml(data_cls: Type[T], data: str) -> T:
    """Instantiate a dataclass from a YAML string.

    Arguments:
        data_cls: A dataclass type.
        data: A YAML string.

    Returns:
        A `data_cls` instance.
    """
    return from_omega(data_cls, OmegaConf.create(data))


def to_yaml(data: DataClass, sort_keys: bool = False) -> str:
    """Serialize a dataclass instance to YAML.

    Arguments:
        data: A dataclass instance.
        sort_keys: Sort mapping keys in YAML output when `True`.

    Returns:
        A YAML string representation of `data`.
    """
    return OmegaConf.to_yaml(to_omega(data), sort_keys=sort_keys)


def from_dotlist(data_cls: Type[T], data: List[str]) -> T:
    """Instantiate a dataclass from a list of dot-style strings.

    Arguments:
        data_cls: A dataclass type.
        data: A list of dot-style strings.
            For example, `["foo.bar=1", "foo.bis=[baz,qux]"]`.

    Returns:
        A `data_cls` instance.
    """
    return from_omega(data_cls, OmegaConf.from_dotlist(data))
