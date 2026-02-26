"""Types and protocols."""

from collections.abc import Sequence
from dataclasses import is_dataclass
from types import UnionType
from typing import Any, ClassVar, Literal, Protocol, Union, get_args
from typing import get_origin as _get_origin


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


def get_origin(t: type) -> type:
    origin = _get_origin(t)
    return t if origin is None else origin


def is_dataclass_instance(x: Any) -> bool:  # noqa: ANN401
    return is_dataclass(type(x))


def is_dataclass_type(x: Any) -> bool:  # noqa: ANN401
    return is_dataclass(x) and not is_dataclass_instance(x)


def is_literal(t: type) -> bool:
    return get_origin(t) is Literal


def is_sequence(t: type) -> bool:
    return issubclass(get_origin(t), Sequence)


def is_union(t: type) -> bool:
    origin = get_origin(t)
    return origin is Union or origin is UnionType


def type_repr(t: type) -> str:
    origin = get_origin(t)
    args = get_args(t)

    if t is Ellipsis:
        return "..."
    elif t is type(None):
        return "None"
    elif is_literal(t):
        return "{" + ", ".join(map(repr, args)) + "}"
    elif is_union(t):
        return "Union[" + ", ".join(map(type_repr, args)) + "]"

    r = getattr(origin, "__name__", repr(origin))

    if args:
        return r + "[" + ", ".join(map(type_repr, args)) + "]"
    else:
        return r
