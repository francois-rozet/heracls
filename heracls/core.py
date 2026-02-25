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

import cattrs
import dataclasses
import yaml

from functools import partial
from omegaconf import DictConfig, OmegaConf
from types import UnionType
from typing import Any, TypeVar, get_args

from .typing import Dataclass, get_origin, is_dataclass_instance, is_literal, is_union, type_repr

DC = TypeVar("DC", bound=Dataclass)


def structure_union(
    conv: cattrs.Converter,
    val: Any,  # noqa: ANN401
    u: UnionType,
) -> Any:  # noqa: ANN401
    matches = []

    unstruct = conv.unstructure(val)

    for t in get_args(u):
        origin = get_origin(t)
        args = get_args(t)

        if t is type(None):
            if val is None:
                matches.append((t, 2, val))
        elif is_literal(t):
            if val in args:
                matches.append((t, 2, val))
        else:
            try:
                if isinstance(val, origin):
                    score = 1
                else:
                    score = 0

                matches.append((t, score, conv.structure(unstruct, t)))
            except Exception:  # noqa: BLE001, S110
                pass

    if len(matches) >= 1:
        best = max(score for _, score, _ in matches)
        matches = [(t, struct) for t, score, struct in matches if score == best]
        if len(matches) == 1:
            return matches[0][1]
        else:
            raise TypeError(f"ambiguous value {val!r} for {type_repr(u)}")
    else:
        raise TypeError(f"incompatible value {val!r} for {type_repr(u)}")


def from_dict(data_cls: type[DC], data: dict[str, Any]) -> DC:
    """Instantiate a dataclass from a dictionary.

    Arguments:
        data_cls: A dataclass type.
        data: A dictionary.

    Returns:
        A `data_cls` instance.
    """
    conv = cattrs.Converter(forbid_extra_keys=True)
    conv.register_structure_hook_func(
        is_union,
        partial(structure_union, conv),
    )

    return conv.structure(data, data_cls)


def to_dict(data: Dataclass, *, recursive: bool = True) -> dict[str, Any]:
    """Convert a dataclass instance to a dictionary.

    Arguments:
        data: A dataclass instance.
        recursive: Recurse into nested dataclasses, dicts, lists, and tuples when `True`.

    Returns:
        A dictionary representation of `data`.
    """
    if not is_dataclass_instance(data):
        return data
    elif recursive:
        return dataclasses.asdict(data)
    keys = [f.name for f in dataclasses.fields(data)]
    data = {k: getattr(data, k) for k in keys}
    return data


def from_omega(data_cls: type[DC], data: DictConfig) -> DC:
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


def from_yaml(data_cls: type[DC], data: str) -> DC:
    """Instantiate a dataclass from a YAML string.

    Arguments:
        data_cls: A dataclass type.
        data: A YAML string.

    Returns:
        A `data_cls` instance.
    """
    return from_dict(data_cls, yaml.safe_load(data))


def to_yaml(data: Dataclass, **kwargs) -> str:  # noqa: ANN003
    """Serialize a dataclass instance to a YAML string.

    Arguments:
        data: A dataclass instance.
        kwargs: Keyword arguments passed to :func:`yaml.safe_dump`.

    Returns:
        A YAML string representation of `data`.
    """
    kwargs.setdefault("sort_keys", False)
    return yaml.safe_dump(to_dict(data, recursive=True), **kwargs)


def from_dotlist(data_cls: type[DC], data: list[str]) -> DC:
    """Instantiate a dataclass from a list of dot-style strings.

    Arguments:
        data_cls: A dataclass type.
        data: A list of dot-style strings.
            For example, `["foo.bar=1", "foo.bis=[baz,qux]"]`.

    Returns:
        A `data_cls` instance.
    """
    return from_omega(data_cls, OmegaConf.from_dotlist(data))
