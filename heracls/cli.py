"""CLI utils and parsing."""

from __future__ import annotations

__all__ = [
    "ArgumentParser",
    "choice",
]

import argparse
import copy
import json

from dataclasses import MISSING, dataclass, field, is_dataclass
from dataclasses import fields as iter_fields
from functools import partial
from typing import (
    Callable,
    Generic,
    Literal,
    TypeVar,
    get_args,
    get_origin,
)
from unittest.mock import patch

from .core import from_dict
from .typing import Dataclass

T = TypeVar("T")
DC = TypeVar("DC", bound=Dataclass)
HELP = "<HELP>"


@dataclass
class ChoiceSpec(Generic[DC]):
    options: dict[str, type[DC]]
    default: str | None = MISSING


@dataclass
class DataclassSpec:
    data_cls: Dataclass
    keys: set[str]
    choices: dict[str, ChoiceSpec]
    root: bool


def boolean(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "on"}


def dict_parser(t: T) -> Callable[[str], T]:
    types = get_args(t)
    types = types if types else (str, str)

    key_parser, value_parser = any_parser(types[0]), any_parser(types[1])

    return lambda s: {key_parser(k): value_parser(v) for k, v in json.loads(s).items()}


def tuple_parser(t: T) -> Callable[[str], T]:
    types = get_args(t)
    types = types if types else (str, Ellipsis)

    if Ellipsis in types:
        parser = any_parser(types[0])
        return lambda s: tuple(parser(x) for x in s.split())
    else:
        parsers = map(any_parser, types)
        return lambda s: tuple(parser(x) for parser, x in zip(parsers, s.split(), strict=True))


def any_parser(t: T) -> Callable[[str], T]:
    if issubclass(t, bool):
        return boolean
    elif issubclass(t, (int, float, str)):
        return t
    elif issubclass(t, dict):
        parser = dict_parser(t)
    elif issubclass(t, tuple):
        parser = tuple_parser(t)
    else:
        return str

    parser.__name__ = t.__name__

    return parser


def choice(
    options: dict[str, type[DC]],
    default: str | None = MISSING,
    default_factory: Callable[[], DC] | None = MISSING,
) -> DC:
    """Create a dataclass field that gives a choice between named options."""

    for option in options.values():
        assert is_dataclass(option) and not is_dataclass(type(option))

    if default_factory is MISSING and default is not MISSING:
        default_factory = options[default]

    return field(
        default_factory=default_factory,
        metadata={"heracls_choice": ChoiceSpec(options, default)},
    )


class SimpleHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter,
):
    def _get_default_metavar_for_optional(self, action: argparse.Action):
        return getattr(action.type, "__name__", None)

    def _get_help_string(self, *args, **kwargs) -> str | None:
        help = super()._get_help_string(*args, **kwargs)
        if isinstance(help, str):
            help = help.replace(HELP, "")
        return help


class ArgumentParser(argparse.ArgumentParser):
    """Simple dataclass-aware argument parser."""

    def __init__(self, *args, **kwargs):
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
        assert is_dataclass(data_cls)
        assert not is_dataclass(type(data_cls))
        assert dest not in self.heracls_specs

        spec = DataclassSpec(
            data_cls=data_cls,
            keys=set(),
            choices={},
            root=root,
        )

        assert dest not in self.heracls_specs

        self.heracls_specs[dest] = spec
        self.register_dataclass(spec, data_cls, () if root else (dest,))

    def register_dataclass(
        self,
        spec: DataclassSpec,
        data_cls: type[Dataclass],
        path: tuple[str, ...] = (),
    ):
        if path:
            group = self.add_argument_group(".".join(path))
        else:
            group = self.add_argument_group()

        for f in iter_fields(data_cls):
            f_path = (*path, f.name)
            f_key = ".".join(f_path)
            f_flag = f"--{f_key}"

            if "heracls_choice" in f.metadata:
                spec.choices[f_key] = f.metadata["heracls_choice"]
                options = f.metadata["heracls_choice"].options
                default = f.metadata["heracls_choice"].default

                group.add_argument(
                    f_flag,
                    dest=f_key,
                    choices=tuple(options.keys()),
                    default=default,
                    help=HELP,
                )
            elif is_dataclass(f.type):
                self.register_dataclass(spec, f.type, f_path)
            else:
                if f.default is MISSING and f.default_factory is MISSING:
                    default = None
                elif f.default is MISSING:
                    default = f.default_factory()
                else:
                    default = f.default

                spec.keys.add(f_key)

                kwargs = {"dest": f_key, "default": default, "help": HELP}

                if get_origin(f.type) == Literal:
                    types = get_args(f.type)
                    group.add_argument(f_flag, choices=types, **kwargs)
                elif f.type is tuple or get_origin(f.type) is tuple:
                    types = get_args(f.type)
                    types = types if types else (str, Ellipsis)

                    if Ellipsis in types:
                        group.add_argument(f_flag, nargs="+", type=any_parser(types[0]), **kwargs)
                    elif all(t == types[0] for t in types):
                        group.add_argument(
                            f_flag, nargs=len(types), type=any_parser(types[0]), **kwargs
                        )
                    else:
                        group.add_argument(f_flag, type=any_parser(f.type), **kwargs)
                elif f.type is list or get_origin(f.type) is list:
                    types = get_args(f.type)
                    types = types if types else (str,)

                    group.add_argument(f_flag, nargs="*", type=any_parser(types[0]), **kwargs)
                elif f.type is dict or get_origin(f.type) is dict:
                    group.add_argument(f_flag, type=any_parser(f.type), **kwargs)
                else:
                    group.add_argument(f_flag, type=any_parser(f.type), **kwargs)

    def finalize(self, args=None, namespace=None) -> ArgumentParser:
        clone = copy.deepcopy(self)
        visited = set()

        while True:
            with patch("argparse._HelpAction.__call__"):
                namespace, _ = argparse.ArgumentParser.parse_known_args(
                    clone, args=args, namespace=namespace
                )

            finished = True
            for spec in clone.heracls_specs.values():
                for key, choice in spec.choices.items():
                    if key not in visited:
                        visited.add(key)
                        value = getattr(namespace, key)
                        clone.register_dataclass(spec, choice.options[value], key.split("."))
                        finished = False

            if finished:
                break

        return clone

    def parse_known_args(self, args=None, namespace=None):
        clone = self.finalize(args, namespace)

        namespace, unknown = argparse.ArgumentParser.parse_known_args(
            clone, args=args, namespace=namespace
        )

        reverse = {}
        for dest, spec in clone.heracls_specs.items():
            for key in spec.keys:
                reverse[key] = dest

        data_dicts = {dest: {} for dest in clone.heracls_specs}
        others = {}

        for key, value in vars(namespace).items():
            if key in reverse:
                data_dicts[reverse[key]][key] = value
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
