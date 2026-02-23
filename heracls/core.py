"""Core data transformations."""

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
import yaml

from dataclasses import fields as iter_fields
from dataclasses import is_dataclass, asdict
from omegaconf import DictConfig, OmegaConf
from typing import Any, ClassVar, Dict, List, Protocol, Type, TypeVar


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


DC = TypeVar("DC", bound=Dataclass)


def from_dict(data_cls: Type[DC], data: Dict[str, Any]) -> DC:
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


def to_dict(data: Dataclass, *, recursive: bool = True) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary.

    Arguments:
        data: A dataclass instance.
        recursive: Recurse into nested dataclasses, dicts, lists, and tuples when `True`.

    Returns:
        A dictionary representation of `data`.
    """
    if not is_dataclass(type(data)):
        return data
    elif recursive:
        return asdict(data)
    keys = [f.name for f in iter_fields(data)]
    data = {k: getattr(data, k) for k in keys}
    return data


def from_omega(data_cls: Type[DC], data: DictConfig) -> DC:
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


def to_omega(data: Dataclass) -> DictConfig:
    """Convert a dataclass instance to an :mod:`omegaconf` config.

    Arguments:
        data: A dataclass instance.

    Returns:
        A config representation of `data`.
    """
    return OmegaConf.create(to_dict(data, recursive=True))


def from_yaml(data_cls: Type[DC], data: str) -> DC:
    """Instantiate a dataclass from a YAML string.

    Arguments:
        data_cls: A dataclass type.
        data: A YAML string.

    Returns:
        A `data_cls` instance.
    """
    return from_dict(data_cls, yaml.safe_load(data))


def to_yaml(data: Dataclass, **kwargs) -> str:
    """Serialize a dataclass instance to a YAML string.

    Arguments:
        data: A dataclass instance.
        kwargs: Keyword arguments passed to :func:`yaml.safe_dump`.

    Returns:
        A YAML string representation of `data`.
    """
    kwargs.setdefault("sort_keys", False)
    return yaml.safe_dump(to_dict(data, recursive=True), **kwargs)


def from_dotlist(data_cls: Type[DC], data: List[str]) -> DC:
    """Instantiate a dataclass from a list of dot-style strings.

    Arguments:
        data_cls: A dataclass type.
        data: A list of dot-style strings.
            For example, `["foo.bar=1", "foo.bis=[baz,qux]"]`.

    Returns:
        A `data_cls` instance.
    """
    return from_omega(data_cls, OmegaConf.from_dotlist(data))
