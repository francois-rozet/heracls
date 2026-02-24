"""CLI utils and parsing."""

from __future__ import annotations

__all__ = [
    "ArgumentParser",
    "choice",
]

import argparse
import copy
import json

from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, is_dataclass
from dataclasses import fields as iter_fields
from functools import partial
from types import UnionType
from typing import (
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from unittest.mock import patch

from .core import from_dict
from .typing import Dataclass

T = TypeVar("T", bound=type)
DC = TypeVar("DC", bound=Dataclass)
HELP = "<HELP>"


@dataclass
class ChoiceSpec(Generic[DC]):
    options: dict[str, type[DC]]
    default: str | None = None


@dataclass
class DataclassSpec:
    data_cls: type[Dataclass]
    keys: set[str]
    choices: dict[str, ChoiceSpec]
    root: bool


def origin(t: T) -> T:
    return t if get_origin(t) is None else get_origin(t)


def boolean(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "on"}


def dict_parser(t: T) -> Callable[[str], T]:
    return lambda s: json.loads(s)


def tuple_parser(t: T) -> Callable[[str], T]:
    types = get_args(t)
    types = types if types else (str, Ellipsis)

    if Ellipsis in types:
        parser = any_parser(types[0])
        return lambda s: tuple(parser(x) for x in s.split())
    else:
        parsers = map(any_parser, types)
        return lambda s: tuple(parser(x) for parser, x in zip(parsers, s.split(), strict=True))


def list_parser(t: T) -> Callable[[str], T]:
    types = get_args(t)
    types = types if types else (str,)

    parser = any_parser(types[0])
    return lambda s: [parser(x) for x in s.split()]


def any_parser(t: T) -> Callable[[str], T]:
    if origin(t) in (int, float, str):
        return origin(t)
    if origin(t) is bool:
        return boolean
    elif origin(t) is dict:
        parser = dict_parser(t)
    elif origin(t) is tuple:
        parser = tuple_parser(t)
    elif origin(t) is list:
        parser = list_parser(t)
    else:
        return str

    parser.__name__ = origin(t).__name__

    return parser


def choice(
    options: dict[str, type[DC]],
    default: str | None = None,
    default_factory: Callable[[], DC] | None = None,
) -> DC:
    """Create a dataclass field that gives a choice between named options."""

    for option in options.values():
        assert is_dataclass(option) and not is_dataclass(type(option))

    if default_factory is None and default is not None:
        default_factory = options[default]

    metadata = {"heracls_choice": ChoiceSpec(options, default)}

    if default_factory is None:
        return field(metadata=metadata)
    else:
        return field(default_factory=default_factory, metadata=metadata)


class SimpleHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter,
):
    def _get_default_metavar_for_optional(self, action: argparse.Action) -> str:
        return getattr(action.type, "__name__", None)

    def _get_help_string(self, *args, **kwargs) -> str | None:  # noqa: ANN002, ANN003
        help = super()._get_help_string(*args, **kwargs)
        if help is not None:
            help = help.replace(HELP, "")
        return help


class ArgumentParser(argparse.ArgumentParser):
    """Simple dataclass-aware argument parser."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        kwargs.setdefault(
            "formatter_class",
            partial(
                SimpleHelpFormatter,
                max_help_position=64,
                width=96,
            ),
        )

        super().__init__(*args, **kwargs)

        self.heracls_specs: dict[str, DataclassSpec] = {}

    def add_arguments(
        self,
        data_cls: type[Dataclass],
        dest: str = "config",
        root: bool = False,
    ) -> None:
        assert is_dataclass(data_cls) and not is_dataclass(type(data_cls))
        assert dest not in self.heracls_specs

        spec = DataclassSpec(
            data_cls=data_cls,
            keys=set(),
            choices={},
            root=root,
        )

        assert dest not in self.heracls_specs

        self.heracls_specs[dest] = spec
        self._register_dataclass(spec, data_cls, () if root else (dest,))

    def _register_dataclass(
        self,
        spec: DataclassSpec,
        data_cls: type[Dataclass],
        path: tuple[str, ...] = (),
    ) -> None:
        if path:
            group = self.add_argument_group(".".join(path))
        else:
            group = self.add_argument_group()

        for f in iter_fields(data_cls):
            f_path = (*path, f.name)
            f_key = ".".join(f_path)
            f_flag = f"--{f_key}"

            kwargs = {"dest": f_key, "help": HELP}

            if "heracls_choice" in f.metadata:
                spec.choices[f_key] = f.metadata["heracls_choice"]

                options = f.metadata["heracls_choice"].options
                default = f.metadata["heracls_choice"].default

                if default is None:
                    kwargs["required"] = True
                else:
                    kwargs["default"] = default

                group.add_argument(f_flag, choices=tuple(options.keys()), **kwargs)
            elif is_dataclass(f.type):
                self._register_dataclass(spec, f.type, f_path)
            else:
                spec.keys.add(f_key)

                if f.default is MISSING and f.default_factory is MISSING:
                    kwargs["required"] = True
                elif f.default is MISSING:
                    kwargs["default"] = f.default_factory()
                else:
                    kwargs["default"] = f.default

                f_type = f.type

                if origin(f_type) == Union or isinstance(f_type, UnionType):
                    types = set(get_args(f_type))
                    types.remove(type(None))
                    if len(types) == 1:
                        f_type = types.pop()

                if origin(f_type) == Literal:
                    types = get_args(f_type)
                    group.add_argument(f_flag, choices=types, **kwargs)
                elif origin(f_type) is tuple:
                    types = get_args(f_type)
                    types = types if types else (str, Ellipsis)

                    if Ellipsis in types:
                        group.add_argument(f_flag, nargs="+", type=any_parser(types[0]), **kwargs)
                    elif all(t == types[0] for t in types):
                        group.add_argument(
                            f_flag, nargs=len(types), type=any_parser(types[0]), **kwargs
                        )
                    else:
                        group.add_argument(f_flag, type=any_parser(f_type), **kwargs)
                elif origin(f_type) is list:
                    types = get_args(f_type)
                    types = types if types else (str,)

                    group.add_argument(f_flag, nargs="*", type=any_parser(types[0]), **kwargs)
                elif origin(f_type) is dict:
                    group.add_argument(f_flag, type=any_parser(f_type), **kwargs)
                else:
                    group.add_argument(f_flag, type=any_parser(f_type), **kwargs)

    def _finalize(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> ArgumentParser:
        clone = copy.deepcopy(self)
        visited = set()

        while True:
            with (
                patch("argparse.ArgumentParser.error"),
                patch("argparse._HelpAction.__call__"),
            ):
                namespace, _ = argparse.ArgumentParser.parse_known_args(
                    clone, args=args, namespace=namespace
                )

            finished = True
            for spec in clone.heracls_specs.values():
                for key, choice in spec.choices.items():
                    if key not in visited:
                        visited.add(key)
                        value = getattr(namespace, key, None)
                        if value is not None:
                            clone._register_dataclass(spec, choice.options[value], key.split("."))
                            finished = False

            if finished:
                break

        return clone

    def parse_known_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        clone = self._finalize(args, namespace)

        namespace, unknown = argparse.ArgumentParser.parse_known_args(
            clone, args=args, namespace=namespace
        )

        dest_map = {}
        choices = set()

        for dest, spec in clone.heracls_specs.items():
            for key in spec.keys:
                dest_map[key] = dest
            choices.update(spec.choices)

        data_dicts = {dest: {} for dest in clone.heracls_specs}
        others = {}

        for key, value in vars(namespace).items():
            if key in dest_map:
                data_dicts[dest_map[key]][key] = value
            elif key in choices:
                pass
            else:
                others[key] = value

        for dest, spec in clone.heracls_specs.items():
            data = {}
            for key, value in data_dicts[dest].items():
                temp = data
                *keys, leaf = key.split(".")
                for k in keys:
                    temp = temp.setdefault(k, {})
                temp[leaf] = value

            others[dest] = from_dict(spec.data_cls, data)

        namespace = argparse.Namespace(**others)

        return namespace, unknown
